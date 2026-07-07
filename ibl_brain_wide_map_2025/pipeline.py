# /// brainset-pipeline
# python-version = "3.10"
# dependencies = [
#   "ray<2.42",
#   "numpy==1.24.4",
#   "ONE-api==3.4.0",
#   "ibllib @ git+https://github.com/int-brain-lab/ibllib.git@2ca4ee2065a9b5baabf7d3ac870839a3a7af9ad3",
#   "scipy==1.15.3",
# ]
# ///

import os

os.environ["ONE_REVISION_LAST_BEFORE"] = "2026-02-23"

import argparse
import logging
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import h5py
import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from ibllib.io.video import get_video_frame, url_from_eid
from one.api import ONE, One
from scipy.ndimage import binary_erosion
from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
    SessionDescription,
    SubjectDescription,
    serialize_fn_map,
)
from torch_brain.pipeline import BrainsetPipeline
from utils import decimate_signal, regularize_timeseries, resample_timeseries

logging.basicConfig(level=logging.INFO)

UnitFilterType = Literal["probe_qc", "firing_rate", "unit_qc"]

parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")
parser.add_argument("--list-sessions", type=str, default=False)
parser.add_argument(
    "--download-first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to download data first. Use --no-download-first to skip.",
)
parser.add_argument(
    "--unit-filter",
    type=str,
    action="append",
    choices=get_args(UnitFilterType),
    default=[],
    help=(
        "Which unit filter(s) to apply. Pass zero or more: "
        "probe_qc, firing_rate, unit_qc. Default is none."
    ),
)
parser.add_argument(
    "--example-frame",
    action="store_true",
    default=False,
    help="Whether to extract and store an example video frame. Disabled by default.",
)

PROBES_WITHOUT_DATA = ["f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c"]

TRIAL_KEYS_REMAPPING = {
    "intervals_0": "start",  # start of trial
    "stimOn_times": "stim_on_time",  # visual stimulus appearance
    "goCueTrigger_times": "go_cue_trigger_time",  # audio command sent
    "goCue_times": "go_cue_time",  # audio sound plays
    "firstMovement_times": "movement_onset_time",  # reaction start
    "response_times": "response_time",  # wheel threshold crossing
    "feedback_times": "feedback_time",  # outcome delivery
    "stimOff_times": "stim_off_time",  # visual stimulus removal
    "intervals_1": "end",  # end of trial
    "probabilityLeft": "probability_left",
    "feedbackType": "feedback_type",
    "rewardVolume": "reward_volume",
    "contrastLeft": "contrast_left",
    "contrastRight": "contrast_right",
}
TRIAL_TIMEKEYS = [
    "start",
    "end",
    "stim_on_time",
    "feedback_time",
    "go_cue_trigger_time",
    "go_cue_time",
    "stim_off_time",
    "response_time",
    "movement_onset_time",
]

# Switch left and right
LEFT_PAW_KEYS = {
    "pos": ["paw_r_x", "paw_r_y"],
    "likelihood": "paw_r_likelihood",
}

RIGHT_PAW_KEYS = {
    "pos": ["paw_l_x", "paw_l_y"],
    "likelihood": "paw_l_likelihood",
}

PRE_STIM_WINDOW = 0.9  # 0.9s before stimulus onset
POST_STIM_WINDOW = 0.1  # 0.1s after stimulus onset
BLOCK_PRIOR_WINDOW = 1.0  # 1s before stimulus onset
PRE_CHOICE_WINDOW = 1.0  # 1s before choice is made
MOVEMENT_WINDOW = 1.0  # 1s after movement starts
FEEDBACK_WINDOW = 1.0  # 1s after reward/penalty

WHISKER_NOMINAL_FS = [60, 150]
PAWS_NOMINAL_FS = [60, 150]

WHEEL_INIT_FS = 1000.0  # 1kHZ
WHEEL_DOWNSAMPLE_FACTORS = [10, 2]  # 1kHZ -> 100Hz -> 50Hz

GAP_TOLERANCE_SEC = 0.01  # 10HZ
TOLERANCE_HZ = 1.0  # 1HZ
TARGET_FS = 50  # 50kHZ

CONTRAST_MAP = {0.0: 0, 0.0625: 1, 0.125: 2, 0.25: 3, 1.0: 4}
REWARD_MAP = {-1: 0, 1: 1}
# In general 1 encode left and 0 right
STIMULUS_SIDE_MAP = {"Left": 1, "Right": 0}
CHOICE_MAP = {-1: 0, 0: 2, 1: 1}
BLOCK_MAP = {0.2: 0, 0.8: 1, 0.5: -1}  # the block with no prior (0.5) will be ignore

# unit filtering parameters
PROBE_QC_LABEL = "PASS"
MIN_FIRING_RATE = 1.0
UNIT_QC_LABEL = 1.0

# region mappings
BRAIN_REGION_ATLASES = ["allen", "beryl", "cosmos"]

# filter trials
MIN_RT = 0.0
MAX_RT = 10.0
EXCLUDE_NO_CHOICE = True
KEYS_WITH_NAN_TO_EXCLUDE = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
]
QUERY = (
    f" | (firstMovement_times - stimOn_times < {MIN_RT})"
    f" | (firstMovement_times - stimOn_times > {MAX_RT})"
)
for event in KEYS_WITH_NAN_TO_EXCLUDE:
    QUERY += f" | {event}.isnull()"
if EXCLUDE_NO_CHOICE:
    QUERY += " | (choice == 0)"
SUCCESSFUL_TRIALS_QUERY = QUERY.lstrip(" |")
ONE_DETAILS_MAX_RETRIES = 3

QC_FILENAME = "bwm_qc.csv"
QC_NEURAL_FIELDS = [
    "qc_neural_raw",
    "qc_neural_spikesorting",
    "qc_neural_alignment",
    "qc_neural",
]
QC_BEHAVIOR_FIELDS = [
    "qc_behavior_whisker",
    "qc_behavior_paws",
    "qc_behavior_wheel",
    "qc_behavior",  # per session, the most severe of the qc_behavior_{signal} metrics above
]

TRAIN_RATIO = 0.4
VAL_RATIO = 0.2
TEST_RATIO = 0.4

MIN_SPLIT_DISTANCE = 20.0  # 20s gap between splits

CONFIDENCE_THRESHOLD = 0.9
CONFIDENCE_MASKS = {
    # maps paws signal name to paws confidence mask name
    "left_paw_v_xy": "is_left_paw_confident",
    "left_paw_speed": "is_left_paw_confident",
    "right_paw_v_xy": "is_right_paw_confident",
    "right_paw_speed": "is_right_paw_confident",
}
INTERVAL_CONFIDENCE_MASKS = {
    # maps task-aligned interval name to paws confidence masks
    "movement_intervals": ["is_left_paw_confident", "is_right_paw_confident"],
}


class Pipeline(BrainsetPipeline):
    brainset_id = "ibl_brain_wide_map_2025"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args):
        pipeline_dir = Path(__file__).resolve().parent
        qc_df = _get_qc_df()

        manifest_list = []
        with open(pipeline_dir / "recording_ids.txt") as fh:
            for line in fh:
                eid = line.strip()
                _qc_df = qc_df[qc_df.eid == eid]
                assert _qc_df.qc_behavior.nunique() == 1
                assert _qc_df.qc_behavior_whisker.nunique() == 1
                assert _qc_df.qc_behavior_paws.nunique() == 1
                assert _qc_df.qc_behavior_wheel.nunique() == 1

                manifest_list.append(
                    {
                        "id": eid,
                        "eid": eid,
                        "qc_behavior": _qc_df.qc_behavior.iloc[0],
                        "qc_behavior_whisker": _qc_df.qc_behavior_whisker.iloc[0],
                        "qc_behavior_paws": _qc_df.qc_behavior_paws.iloc[0],
                        "qc_behavior_wheel": _qc_df.qc_behavior_wheel.iloc[0],
                    }
                )

        manifest = pd.DataFrame(manifest_list).set_index("id")

        # connect to the ONE sdk
        # do this once only, and copy the handle for all manifest items
        # otherwise, an error will be raised from token file being overwritten
        ONE.setup(silent=True)
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            username="intbrainlab",
            password="international",
            cache_dir=raw_dir,
        )
        manifest["one"] = one

        return manifest

    def download(self, manifest_item):
        if not self.args.download_first:
            self.update_status("Skipped Downloading (--no-download-first flag used)")
            # We still MUST return manifest_item so the process() method
            # has the eid and ONE instance to work with from the local cache!
            return manifest_item

        self.update_status("Starting Download")

        retries = 0
        while retries < ONE_DETAILS_MAX_RETRIES:
            try:
                self._download_from_one(manifest_item)
                break
            except Exception as e:
                retries += 1
                if retries == ONE_DETAILS_MAX_RETRIES:
                    raise Exception(
                        f"Failed to download from ONE after {ONE_DETAILS_MAX_RETRIES} attempts"
                    ) from e
                self.update_status(
                    f"Failed to download from ONE on attempt {retries}. Waiting before retrying..."
                )
                logging.warning(f"Failed to download from ONE on attempt {retries}: {e}")
                time.sleep(5)  # wait 5 seconds before retrying

        self.update_status("Ending Download")

        return manifest_item

    def _download_from_one(self, manifest_item):
        eid = manifest_item.eid
        one: One = manifest_item.one

        self.update_status("Downloading Spikes")
        pids, probe_names = one.eid2pid(eid)
        for pid, probe_name in zip(pids, probe_names, strict=True):
            if str(pid) in PROBES_WITHOUT_DATA:
                logging.warning(f"Probe {pid} has no data, skipping")
                continue
            # mock=True skips the ~37 MB Allen volumetric atlas download (template +
            # annotation .nrrd) which the pipeline never uses; only atlas.regions is
            # consumed, to map already-resolved channel IDs to acronyms.
            spike_loader = SpikeSortingLoader(
                pid=pid, one=one, eid=eid, pname=probe_name, atlas=AllenAtlas(mock=True)
            )
            spike_loader.download_spike_sorting()

        one_details = one.get_details(eid)
        local_path = one_details["local_path"]
        session_loader = SessionLoader(one=one, session_path=local_path, eid=eid)

        self.update_status("Downloading Trial")
        session_loader.load_trials()

        if manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Whisker")
            session_loader.load_motion_energy(views=["left"])

        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Pose")
            session_loader.load_pose(likelihood_thr=0.0, views=["left"], tracker="lightningPose")

        if manifest_item.qc_behavior_wheel not in ["FAIL", "MISSING"]:
            self.update_status("Downloading Wheel")
            session_loader.load_wheel()

    def process(self, manifest_item):
        eid = manifest_item.eid
        one: One = manifest_item.one

        store_dir = self.processed_dir
        store_dir.mkdir(exist_ok=True, parents=True)

        store_path = store_dir / f"{eid}.h5"

        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Processing session")

        brainset_description = BrainsetDescription(
            id="ibl_brain_wide_map_2025",
            origin_version="0.0.1",
            derived_version="0.0.5",
            source="one-api",
            description=(
                "A key challenge in neuroscience is understanding how neurons in hundreds of interconnected brain regions integrate sensory inputs with "
                "previous expectations to initiate movements and make decisions. It is difficult to meet this challenge if different laboratories apply "
                "different analyses to different recordings in different regions during different behaviours. Here we report a comprehensive set of recordings "
                "from 621,733 neurons recorded with 699 Neuropixels probes across 139 mice in 12 laboratories. The data were obtained from mice performing "
                "a decision-making task with sensory, motor and cognitive components. The probes covered 279 brain areas in the left forebrain and midbrain "
                "and the right hindbrain and cerebellum. We provide an initial appraisal of this brain-wide map and assess how neural activity encodes key "
                "task variables. Representations of visual stimuli transiently appeared in classical visual areas after stimulus onset and then spread to "
                "ramp-like activity in a collection of midbrain and hindbrain regions that also encoded choices. Neural responses correlated with impending "
                "motor action almost everywhere in the brain. Responses to reward delivery and consumption were also widespread. This publicly available "
                "dataset represents a resource for understanding how computations distributed across and within brain areas drive behaviour."
            ),
        )

        # get subject metadata
        one_details = one.get_details(eid)
        subject_id = one_details["subject"]

        subject = SubjectDescription(
            id=subject_id,
            species="MUS_MUSCULUS",
        )

        # extract experiment metadata
        recording_date = datetime.fromisoformat(one_details["start_time"]).strftime("%Y%m%d")
        lab_id = one_details["lab"]

        device_id = f"{lab_id}_{subject.id}_{recording_date}"

        # register session
        session_description = SessionDescription(
            id=eid,
            recording_date=datetime.strptime(recording_date, "%Y%m%d"),
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech="NEUROPIXELS_ARRAY",
        )

        # extract spiking activity
        self.update_status("Extracting Spikes")
        spikes, units, probes = extract_spikes(one, eid, unit_filters=self.args.unit_filter)

        # extract behavioral data
        self.update_status("Extracting Behavioral Data")

        session_loader = SessionLoader(one=one, session_path=one_details["local_path"], eid=eid)

        behavior_dict = {}

        if manifest_item.qc_behavior == "PASS":
            assert (
                manifest_item.qc_behavior_whisker == "PASS"
                and manifest_item.qc_behavior_paws == "PASS"
                and manifest_item.qc_behavior_wheel == "PASS"
            ), "If behavior QC is PASS, then all other specific behavior QC must be PASS"

        ref_resampled_timestamps = None
        # whisker motion energ
        if manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]:
            # whisker data is always available and all other behavior timestamps will be aligned to it
            self.update_status("Extracting Whisker Data")
            whisker = extract_whisker(session_loader)
            ref_resampled_timestamps = whisker.timestamps

            validate_whisker(whisker, ref_fs=TARGET_FS)
            behavior_dict["whisker"] = whisker

        else:
            logging.warning("Whisker data is skipped because of QC")

        # paws
        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]:
            self.update_status("Extracting Pose Data")
            paws = extract_paws(session_loader)

            if ref_resampled_timestamps is None:
                ref_resampled_timestamps = paws.timestamps

            if paws is not None:
                validate_paws(
                    paws,
                    ref_fs=TARGET_FS,
                    ref_timestamps=ref_resampled_timestamps,
                )
                behavior_dict["paws"] = paws
            else:
                logging.warning("Pose data is not available")

        else:
            logging.warning("Pose data is skipped because of QC")

        # wheel
        if (
            manifest_item.qc_behavior_wheel not in ["FAIL", "MISSING"]
            and ref_resampled_timestamps is not None
        ):
            self.update_status("Extracting Wheel Data")
            wheel = extract_wheel(session_loader, ref_resampled_timestamps=ref_resampled_timestamps)
            if wheel is not None:
                validate_wheel(
                    wheel,
                    ref_fs=TARGET_FS,
                    ref_timestamps=ref_resampled_timestamps,
                )
                behavior_dict["wheel"] = wheel
            else:
                logging.warning("Wheel data is not available")

        else:
            logging.warning("Wheel data skipped because of QC")

        if self.args.example_frame and (
            manifest_item.qc_behavior_whisker not in ["FAIL", "MISSING"]
            or manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"]
        ):
            self.update_status("Extracting Example Video Frame")
            behavior_dict["video"] = extract_video(eid, one)

        # register behavior QC
        self.update_status("Registering Behavior QC")
        behavior_qc = get_beh_qc(manifest_item)

        # extract trial data
        self.update_status("Extracting Trial and Task Intervals Data")
        trials = load_trials(session_loader)

        # extract task-aligned intervals
        task_aligned_intervals = extract_task_aligned_intervals(trials)
        if manifest_item.qc_behavior_paws not in ["FAIL", "MISSING"] and paws is not None:
            task_aligned_intervals = _fill_interval_confidence_counts(task_aligned_intervals, paws)

        # register session
        # Note: it is possible to have a bigger data.domain than spikes.domain
        # as spikes.domain is not necessarly a superset of all the behaviors' domains.
        # We decide to restrain splits and data.domain to the spikes domain for the brain wide map
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            lab=lab_id,
            # neural activity
            spikes=spikes,
            units=units,
            probes=probes,
            # stimuli and behavior
            trials=trials,
            task_aligned_intervals=task_aligned_intervals,
            **behavior_dict,
            behavior_qc=behavior_qc,
            domain=spikes.domain,
        )

        # define splits trial data
        self.update_status("Extracting Splits")

        # Splits balance trial COUNT across train/val/test (equal sample budgets).
        train_domain, val_domain, test_domain = get_causal_splits(
            data.domain,
            task_aligned_intervals.domain,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
            MIN_SPLIT_DISTANCE,
            split_by="count",
        )

        data.train_domain = train_domain
        data.val_domain = val_domain
        data.test_domain = test_domain

        data.normalize = _get_beh_normalize(data, train_domain)

        # save data to disk
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        try:
            one.save_cache()  # explicitly save before shutdown
        except Exception as e:
            logging.warning(f"Error saving cache: {e}")


def _get_beh_normalize(data: Data, train_domain: Interval):
    """Compute behavior normalization stats using only train-split movement intervals."""

    eps = 1e-8
    normalize = Data()

    movement_intervals = data.task_aligned_intervals.movement_intervals
    intervals = movement_intervals & train_domain

    specs = {
        "whisker": ["motion_energy"],
        "paws": [
            "right_paw_v_xy",
            "left_paw_v_xy",
            "right_paw_speed",
            "left_paw_speed",
        ],
    }

    for modality_name, signal_names in specs.items():
        modality_norm = Data()

        for signal_name in signal_names:
            signal_values = []
            for s, e in intervals:
                interval_slice = data.slice(s, e, reset_origin=False)
                modality = getattr(interval_slice, modality_name, None)
                if modality is None or not hasattr(modality, signal_name):
                    continue
                values = _get_normalization_values(modality, modality_name, signal_name)
                if values.size > 0:
                    signal_values.append(values)

            if signal_values:
                values = np.concatenate(signal_values, axis=0)
                setattr(
                    modality_norm,
                    signal_name,
                    Data(
                        mean=values.mean(axis=0),
                        std=np.maximum(values.std(axis=0), eps),
                    ),
                )

        setattr(normalize, modality_name, modality_norm)

    return normalize


def _get_normalization_values(
    modality: RegularTimeSeries,
    modality_name: str,
    signal_name: str,
) -> np.ndarray:
    values = np.asarray(getattr(modality, signal_name))
    if modality_name != "paws":
        return values

    mask_name = CONFIDENCE_MASKS.get(signal_name)
    if mask_name is None or not hasattr(modality, mask_name):
        return values

    mask = np.asarray(getattr(modality, mask_name), dtype=bool)
    assert mask.shape[0] == values.shape[0], (
        f"Skipping paws confidence mask for {signal_name} due to shape mismatch ({values.shape} vs {mask.shape})."
    )

    masked_values = values[mask]
    return values if masked_values.size == 0 else masked_values


def _get_qc_df(eid: str | None = None) -> pd.DataFrame:
    """Get QC metrics as a dataframe

    Args:
        eid: If given, return only the metadata for that eid.
            If None, return the entire dataframe
    """
    pipeline_dir = Path(__file__).resolve().parent
    qc_df = pd.read_csv(pipeline_dir / QC_FILENAME)

    if eid is None:
        return qc_df

    return qc_df[qc_df.eid == eid]


def _fill_interval_confidence_counts(task_aligned_intervals: Interval, paws: IrregularTimeSeries):
    """For each task-aligned interval defined in INTERVAL_CONFIDENCE_MASKS, count how many
    confident paw samples fall within each interval window and store the result as a new
    attribute (e.g. ``is_left_paw_confident_count``) on the corresponding sub-interval.
    """
    for task_aligned_name, confidence_names in INTERVAL_CONFIDENCE_MASKS.items():
        task_aligned_interval = getattr(task_aligned_intervals, task_aligned_name)
        for confidence_name in confidence_names:
            counts_key = f"{confidence_name}_count"
            confidence_counts = [
                getattr(paws.slice(start, end), confidence_name).sum()
                for start, end in task_aligned_interval
            ]
            setattr(
                task_aligned_interval,
                counts_key,
                np.array(confidence_counts),
            )
    return task_aligned_intervals


def extract_spikes(one: One, eid: str, unit_filters: list[UnitFilterType]):
    """Load and reindex spike-sorting outputs across probes for one session."""

    # Resolve probe insertion IDs and names for this experiment session.
    pids, probe_names = one.eid2pid(eid)

    spikes_list, units_list, templates_list = [], [], []
    unit_ptr = 0

    probe_qc_df = _get_qc_df(eid).set_index("probe_id")

    # Load per-probe spikes and clusters, then concatenate them into
    # a session-level spike stream with globally unique unit indices.
    for pid, probe_name in zip(pids, probe_names, strict=True):
        # mock=True skips the unused ~37 MB Allen volumetric atlas download; only
        # atlas.regions is needed to map resolved channel IDs to acronyms.
        spike_loader = SpikeSortingLoader(
            pid=pid, one=one, eid=eid, pname=probe_name, atlas=AllenAtlas(mock=True)
        )
        spikes, clusters, channels = spike_loader.load_spike_sorting()

        if len(clusters) == 0:
            # A small number of known probe insertions are expected to have no cluster output;
            # this guards against silently dropping unexpected cases.
            assert str(pid) in PROBES_WITHOUT_DATA, f"Pid {pid} has no clusters"
            continue

        # Merge cluster metadata for unit-level annotations.
        clusters = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False
        )

        # Extract waveform templates
        waveforms = spike_loader.raw_waveforms()
        templates = waveforms.templates[clusters.cluster_id]
        templates_list.append(templates)

        # Offset probe-local cluster IDs so unit indices are unique across probes.
        spikes["clusters"] += unit_ptr
        spikes_list.append(spikes)

        clusters["probe_id"] = str(pid)
        for field in QC_NEURAL_FIELDS:
            clusters[field] = str(probe_qc_df.loc[str(pid)][field])

        for atlas in BRAIN_REGION_ATLASES:
            clusters[f"region_{atlas}"] = BrainRegions().acronym2acronym(
                clusters["acronym"], mapping=atlas.capitalize()
            )

        units_list.append(pd.DataFrame(clusters).rename(columns={"uuids": "id"}))

        # Advance index offset for the next probe.
        num_units = len(clusters["cluster_id"])
        unit_ptr += num_units

        assert num_units == len(np.unique(clusters["cluster_id"])), "There are duplicate units"

        assert num_units == len(np.unique(spikes["clusters"])), (
            "There are units that have no spikes"
        )

    timestamps = np.concatenate([s["times"] for s in spikes_list])
    unit_index = np.concatenate([s["clusters"] for s in spikes_list])
    amplitude = np.concatenate([s["amps"] for s in spikes_list])
    depth = np.concatenate([s["depths"] for s in spikes_list])

    # Store units.
    units = ArrayDict.from_dataframe(
        pd.concat(units_list, ignore_index=True),
        unsigned_to_long=True,
    )
    units.wf_template = np.concatenate(templates_list, axis=0)
    units.location = np.stack([units.x, units.y, units.z], axis=1)

    # Store probes.
    probes_dict = defaultdict(list)
    for pid, probe_name in zip(pids, probe_names, strict=True):
        pid = str(pid)
        if pid in PROBES_WITHOUT_DATA:
            continue
        probes_dict["id"].append(pid)
        probes_dict["name"].append(probe_name)
        for field in QC_NEURAL_FIELDS:
            probes_dict[field] = str(probe_qc_df.loc[pid][field])

    probes_df = pd.DataFrame(probes_dict)
    probes = ArrayDict.from_dataframe(probes_df, unsigned_to_long=True)

    # Apply unit filters
    num_total_units = len(units)
    unit_mask = np.ones(num_total_units, dtype=bool)
    if "probe_qc" in unit_filters:
        unit_mask &= units.qc_neural == PROBE_QC_LABEL
    if "firing_rate" in unit_filters:
        unit_mask &= units.firing_rate > MIN_FIRING_RATE
    if "unit_qc" in unit_filters:
        unit_mask &= units.label == UNIT_QC_LABEL
    num_valid_units = unit_mask.sum()
    logging.info(f"Keeping {num_valid_units} units out of {num_total_units} after unit filters")

    # filter units
    units = units.select_by_mask(unit_mask)

    # remap unit indices
    valid_unit_idx = np.where(unit_mask)[0]
    unit_remap = np.zeros(num_total_units, dtype=int)
    unit_remap[unit_mask] = np.arange(num_valid_units)

    # filter spikes
    spike_mask = np.isin(unit_index, valid_unit_idx)
    timestamps = timestamps[spike_mask]
    unit_index = unit_remap[unit_index[spike_mask]]
    amplitude = amplitude[spike_mask]
    depth = depth[spike_mask]

    # Store spikes and sort the timestamps.
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        amplitude=amplitude,
        depth=depth,
        domain="auto",
    )
    spikes.sort()

    # Unit UUIDs should be globally unique after concatenation.
    assert len(units.id) == len(np.unique(units.id)), "uuids is not unique"

    return spikes, units, probes


def get_beh_qc(manifest_item: NamedTuple):
    """Construct an object to store behavior QC from manifest item"""

    beh_qc = Data(**{qc_field: getattr(manifest_item, qc_field) for qc_field in QC_BEHAVIOR_FIELDS})

    return beh_qc


def extract_whisker(session_loader: SessionLoader, return_full: bool = False):
    """Load whisker motion energy, regularize timestamps, and resample to the target fs."""

    session_loader.load_motion_energy(views=["left"])

    df_motion_energy = session_loader.motion_energy["leftCamera"]
    raw_timestamps = df_motion_energy.times.values
    motion_energy = df_motion_energy.whiskerMotionEnergy.values

    assert len(raw_timestamps) == len(motion_energy), (
        f"Number of timestamps ({len(raw_timestamps)}) doesn't match number of motion energy values ({len(motion_energy)})"
    )

    whisker = IrregularTimeSeries(
        timestamps=raw_timestamps, motion_energy=motion_energy, domain="auto"
    )

    # First, regularize the timestamp grid.
    regularized_whisker = regularize_timeseries(whisker, gap_tol=GAP_TOLERANCE_SEC)
    whisker_fs = 1 / np.median(np.diff(regularized_whisker.timestamps))

    # The source video is expected at 60 Hz or 150 Hz.
    assert any(np.abs(whisker_fs - fs) < TOLERANCE_HZ for fs in WHISKER_NOMINAL_FS), (
        f"Whisker sampling fs is not one of {WHISKER_NOMINAL_FS}Hz (or within {TOLERANCE_HZ}Hz), it is {whisker_fs}"
    )

    # Then resample to the shared target fs (50 Hz).
    resampled_whisker = resample_timeseries(regularized_whisker, target_fs=TARGET_FS)

    # Return the resampled trace and raw timestamps for cross-modal alignment.
    if not return_full:
        return resampled_whisker

    return {
        "raw": whisker,
        "regularized": regularized_whisker,
        "resampled": resampled_whisker,
        "regularized_fs": whisker_fs,
        "resampled_fs": TARGET_FS,
    }


def extract_video(eid: str, one: One) -> Data:
    """Extract an example frame from the left camera and return it in a Data object."""
    example_frame = None
    try:
        video_url = url_from_eid(eid, one=one)["left"]
        for retry in range(ONE_DETAILS_MAX_RETRIES):
            try:
                example_frame = get_video_frame(video_url, frame_number=1000)[:, :, 0]
                break
            except Exception:
                if retry < ONE_DETAILS_MAX_RETRIES:
                    logging.info(f"Streaming left video failed, retrying x{retry + 1}")
                    time.sleep(10)
                else:
                    logging.warning("Could not load video frame for left cam.")
    except KeyError:
        logging.warning("Could not load video frame for left cam.")

    return Data(example_frame=example_frame)


def extract_paws(
    session_loader: SessionLoader,
    return_full: bool = False,
):
    """Load paws keypoints, regularize timestamps, and resample to the target fs."""
    session_loader.load_pose(likelihood_thr=0.0, views=["left"], tracker="lightningPose")
    pose_df = session_loader.pose["leftCamera"]
    raw_timestamps = pose_df.times.values

    if pose_df is None:
        return None

    # Extract paw keypoints (exclude nose, pupil, tube and tongue keypoints).
    # The tube is static, and tongue dynamics are represented by lick events.
    # Build paws time series on the video timestamp axis.
    paws = IrregularTimeSeries(
        timestamps=raw_timestamps,
        # paws
        left_paw=pose_df[LEFT_PAW_KEYS["pos"]].to_numpy(),
        left_paw_likelihood=pose_df[LEFT_PAW_KEYS["likelihood"]].to_numpy(),
        right_paw=pose_df[RIGHT_PAW_KEYS["pos"]].to_numpy(),
        right_paw_likelihood=pose_df[RIGHT_PAW_KEYS["likelihood"]].to_numpy(),
        domain="auto",
    )

    # First, regularize both streams to a nearly uniform timestamp grid.
    regularized_paws = regularize_timeseries(paws, gap_tol=GAP_TOLERANCE_SEC)
    paws_fs = 1.0 / np.median(np.diff(regularized_paws.timestamps))

    # Validate that observed fss match expected nominal frame fss.
    assert any(np.abs(paws_fs - fs) < TOLERANCE_HZ for fs in PAWS_NOMINAL_FS), (
        f"Paw sampling fs is not one of {PAWS_NOMINAL_FS}Hz (or within {TOLERANCE_HZ}Hz), it is {paws_fs}"
    )

    # Derive per-axis velocity for key tracked keypoints.
    dt = 1 / paws_fs
    regularized_paws.left_paw_v_xy = np.gradient(regularized_paws.left_paw, dt, axis=0)
    regularized_paws.left_paw_speed = np.linalg.norm(regularized_paws.left_paw_v_xy, axis=1)
    regularized_paws.right_paw_v_xy = np.gradient(regularized_paws.right_paw, dt, axis=0)
    regularized_paws.right_paw_speed = np.linalg.norm(regularized_paws.right_paw_v_xy, axis=1)

    # Resample both streams to the common 50 Hz target fs.
    resampled_paws = resample_timeseries(regularized_paws, target_fs=TARGET_FS)

    left_mask = resampled_paws.left_paw_likelihood >= CONFIDENCE_THRESHOLD
    right_mask = resampled_paws.right_paw_likelihood >= CONFIDENCE_THRESHOLD
    # need to erode the mask as the np.gradient use central differences (Second-Order Accurate)
    resampled_paws.is_left_paw_confident = binary_erosion(left_mask, border_value=1)
    resampled_paws.is_right_paw_confident = binary_erosion(right_mask, border_value=1)

    if not return_full:
        return resampled_paws

    return {
        "raw": paws,
        "regularized": regularized_paws,
        "resampled": resampled_paws,
        "regularized_fs": paws_fs,
        "resampled_fs": TARGET_FS,
    }


def extract_wheel(
    session_loader: SessionLoader,
    ref_resampled_timestamps: np.ndarray,
    return_full: bool = False,
):
    """Load wheel data, regularize timestamps, and resample to the target fs."""

    session_loader.load_wheel()

    timestamps = session_loader.wheel["times"].to_numpy().astype(np.float64)
    pos = session_loader.wheel["position"].to_numpy()
    vel = session_loader.wheel["velocity"].to_numpy()

    wheel = IrregularTimeSeries(timestamps=timestamps, pos=pos, vel=vel, domain="auto")

    # Before resampling the wheel data, it need to align timestamps to the target timestamps

    def snap_interval_round(timestamps, target_fs, t0):
        dt = 1.0 / target_fs
        # Push start forward to the next grid point
        start = t0 + np.ceil((timestamps[0] - t0) / dt) * dt
        # Pull end back to the previous grid point
        end = t0 + np.floor((timestamps[-1] - t0) / dt) * dt
        return start, end

    start_timestamp, end_timestamp = snap_interval_round(
        timestamps, TARGET_FS, ref_resampled_timestamps[0]
    )

    regularized_wheel = regularize_timeseries(
        wheel,
        gap_tol=GAP_TOLERANCE_SEC,
        target_fs=WHEEL_INIT_FS,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    # Decimate from 1kHz to 100Hz, then 100Hz to 50Hz
    pos = decimate_signal(regularized_wheel.pos, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS)
    vel = decimate_signal(regularized_wheel.vel, WHEEL_DOWNSAMPLE_FACTORS, WHEEL_INIT_FS)

    domain_start = regularized_wheel.timestamps[0]
    resampled_wheel = RegularTimeSeries(
        sampling_rate=TARGET_FS,
        domain_start=domain_start,
        pos=pos,
        vel=vel,
        speed=np.abs(vel),
    )

    if not return_full:
        return resampled_wheel

    return {
        "raw": wheel,
        "regularized": regularized_wheel,
        "resampled": resampled_wheel,
        "regularized_fs": WHEEL_INIT_FS,
        "resampled_fs": TARGET_FS,
    }


def load_trials(session_loader: SessionLoader):
    """Load trials data, remap columns, convert to Interval objects, and filter trials."""

    session_loader.load_trials()
    trials = session_loader.trials

    trials = trials.rename(columns=TRIAL_KEYS_REMAPPING)

    # relevant columns: choice, probability_left, ...
    trials = Interval.from_dataframe(trials, timekeys=TRIAL_TIMEKEYS)

    # choice is -1, 0 or 1, we will map it to 0, 2 and 1 respectively
    trials.choice = pd.Series(trials.choice).map(CHOICE_MAP).to_numpy()

    # feedback -1 is 0 and feedback 1 is 1
    trials.reward = pd.Series(trials.feedback_type).map(REWARD_MAP).to_numpy()

    # block is 0.2, 0.5 or 0.8, we will map it to 0, 1 and 2 respectively
    trials.block = pd.Series(trials.probability_left).map(BLOCK_MAP).to_numpy()

    # 0 if right, 1 if left
    side_labels = np.where(np.isnan(trials.contrast_left), "Right", "Left")
    trials.stimulus_side = pd.Series(side_labels).map(STIMULUS_SIDE_MAP).to_numpy()

    # sanity check: the encoded task variables are compatible
    assert np.all(
        # rewarded iff choice is not no choice (0) and choice is the same as stimulus side
        ((trials.choice != CHOICE_MAP[0]) & (trials.choice == trials.stimulus_side))
        == trials.reward
    ), "Stimulus side and contrast are not compatible"

    # left and right contrasts are 0., 0.0625, 0.125, 0.25, 1.0
    contrast = np.nan_to_num(trials.contrast_left) + np.nan_to_num(trials.contrast_right)
    trials.stimulus_contrast = pd.Series(contrast).map(CONTRAST_MAP).to_numpy()

    # filter trials
    trials.successful = ~session_loader.trials.eval(SUCCESSFUL_TRIALS_QUERY).values

    assert trials.is_disjoint(), "Trials are not disjoint"
    trials.sort()

    # trial index
    trials.idx = np.arange(len(trials))

    return trials


def extract_task_aligned_intervals(trials: Interval):
    """Extract task-aligned intervals from trials data."""

    trials_idx = np.arange(len(trials))
    trials_idx = trials_idx[trials.successful]
    trials = trials.select_by_mask(trials.successful)

    # stimulus side
    stimulus_side = Interval(
        start=trials.stim_on_time - PRE_STIM_WINDOW,
        end=trials.stim_on_time + POST_STIM_WINDOW,
        stimulus_side=trials.stimulus_side,
        trials_idx=trials_idx,
    )
    # stimulus contrast
    stimulus_contrast = Interval(
        start=trials.stim_on_time - PRE_STIM_WINDOW,
        end=trials.stim_on_time + POST_STIM_WINDOW,
        stimulus_contrast=trials.stimulus_contrast,
        trials_idx=trials_idx,
    )

    # choice
    choice = Interval(
        start=trials.movement_onset_time - PRE_CHOICE_WINDOW,
        end=trials.movement_onset_time,
        choice=trials.choice,
        trials_idx=trials_idx,
    )

    # reward
    reward = Interval(
        start=trials.feedback_time,
        end=trials.feedback_time + FEEDBACK_WINDOW,
        reward=trials.reward,
        trials_idx=trials_idx,
    )

    # movement intervals
    movement_intervals = Interval(
        start=trials.movement_onset_time,
        end=trials.movement_onset_time + MOVEMENT_WINDOW,
        trials_idx=trials_idx,
    )

    # block prior (we ignore the first block unbiased)
    block_prior = Interval(
        start=trials.stim_on_time - BLOCK_PRIOR_WINDOW,
        end=trials.stim_on_time,
        block=trials.block,
        trials_idx=trials_idx,
    )
    block_prior_mask = block_prior.block != BLOCK_MAP[0.5]
    block_prior = block_prior.select_by_mask(block_prior_mask)

    # sanity check: all the block with no prior are ignore (0.5 block prior)
    assert np.all(block_prior.block != BLOCK_MAP[0.5]), "Some blocks have a prior of 0.5"

    tasks = [
        stimulus_side,
        stimulus_contrast,
        choice,
        movement_intervals,
        reward,
    ]

    # Get the length from the first valid attribute
    num_indices = len(tasks[0])
    assert all(len(t) == num_indices for t in tasks), "Tasks have mismatched lengths"
    assert len(block_prior_mask) == num_indices, "Block prior mask has mismatched length"

    domain_starts = np.full(num_indices, np.inf)
    domain_ends = np.full(num_indices, -np.inf)

    for task in tasks:
        # Using [:] to force loading from HDF5 into memory
        domain_starts = np.minimum(domain_starts, task.start[:])
        domain_ends = np.maximum(domain_ends, task.end[:])

    domain_starts[block_prior_mask] = np.minimum(
        domain_starts[block_prior_mask], block_prior.start[:]
    )
    domain_ends[block_prior_mask] = np.maximum(domain_ends[block_prior_mask], block_prior.end[:])

    if np.any(domain_starts == np.inf) or np.any(domain_ends == -np.inf):
        raise AssertionError("Found trials with NO task-aligned data")

    domain = Interval(start=domain_starts, end=domain_ends)

    assert domain.is_disjoint(), "task_aligned_intervals domain is not disjoint"

    task_aligned_intervals = Data(
        block_prior=block_prior,
        stimulus_side=stimulus_side,
        stimulus_contrast=stimulus_contrast,
        choice=choice,
        reward=reward,
        movement_intervals=movement_intervals,
        domain=domain,
    )

    return task_aligned_intervals


def get_causal_splits(
    domain: Interval,
    task_aligned_domain: Interval,
    train_ratio: float = 0.4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.4,
    min_split_distance: float = 0.0,
    split_by: Literal["time", "count"] = "time",
):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "split are not summing to one"

    assert min_split_distance >= 0.0, "min_split_distance must be non-negative"
    assert split_by in ("time", "count"), "split_by must be 'time' or 'count'"

    start, end = domain.start[0], domain.end[-1]

    # Restraining task_aligned_domain to be a subset of domain
    task_aligned_domain = task_aligned_domain & domain

    task_starts, task_ends = task_aligned_domain.start, task_aligned_domain.end
    task_lengths = task_ends - task_starts

    # Weights decide WHERE the split boundaries land. "time" gives each split an
    # equal share of task-seconds (legacy); because late trials are longer, that
    # skews trial COUNTS toward train. "count" weights every trial equally, so the
    # ratios apply to the number of trials (balanced sample budgets per split).
    # Everything below is unit-agnostic: it locates a boundary in cumulative-weight
    # space and then snaps to a real task-interval edge.
    weights = task_lengths if split_by == "time" else np.ones_like(task_lengths)
    total_weight = float(np.sum(weights))
    cumulative_weight = np.cumsum(weights)

    def _timestamp_from_task_offset(offset: float) -> float:
        """
        Map a cumulative-weight offset (task-seconds or trial count, per
        `split_by`) to a real timestamp, snapped to the nearest task interval
        edge (never inside an interval).
        """
        offset = float(np.clip(offset, 0.0, total_weight))
        idx = int(np.searchsorted(cumulative_weight, offset, side="left"))
        idx = min(idx, len(cumulative_weight) - 1)
        prev = 0.0 if idx == 0 else float(cumulative_weight[idx - 1])
        next_ = float(cumulative_weight[idx])

        # Snap to closest edge in cumulative-time space.
        if (offset - prev) <= (next_ - offset):
            return float(task_starts[idx])
        return float(task_ends[idx])

    def _boundary_with_gap(
        boundary: float,
        min_gap: float,
        task_starts: np.ndarray,
        task_ends: np.ndarray,
        task_edges: np.ndarray,
        label: str,
        eps: float = 1e-12,
    ) -> float:
        """
        Advance a boundary by min_gap while ensuring it does not land inside
        any task interval. If it does, snap to the next interval edge.
        """
        gap_end = boundary + min_gap
        if np.any((task_starts < gap_end) & (gap_end < task_ends)):
            boundary_gap = float(task_edges[task_edges >= gap_end][0])
        else:
            boundary_gap = float(gap_end) + eps

        assert not np.any((task_starts < boundary_gap) & (boundary_gap < task_ends)), (
            f"{label} {boundary_gap} is inside a task interval"
        )
        assert boundary_gap >= boundary, f"{label} {boundary_gap} is not >= boundary {boundary}"
        assert boundary_gap - boundary >= min_gap, (
            f"{label} - boundary {boundary_gap - boundary} is not at least min_gap {min_gap}"
        )
        return boundary_gap

    task_edges = np.sort(np.concatenate([task_starts, task_ends]))

    # get starting boundaries for train and val
    b1 = _timestamp_from_task_offset(total_weight * train_ratio)
    b2 = _timestamp_from_task_offset(total_weight * (train_ratio + val_ratio))

    # Boundary for Validation
    b1_val = _boundary_with_gap(
        boundary=b1,
        min_gap=min_split_distance,
        task_starts=task_starts,
        task_ends=task_ends,
        task_edges=task_edges,
        label="b1_val",
    )

    # create domains
    train_domain = Interval(start, b1)

    if test_ratio > 0:
        # Boundary for Test
        b2_test = _boundary_with_gap(
            boundary=b2,
            min_gap=min_split_distance,
            task_starts=task_starts,
            task_ends=task_ends,
            task_edges=task_edges,
            label="b2_test",
        )

        val_domain = Interval(b1_val, b2)
        test_domain = Interval(b2_test, end)
    else:
        val_domain = Interval(b1_val, end)
        test_domain = Interval(np.array([]), np.array([]))

    # leakage checks
    assert len(train_domain & val_domain) == 0, "Leakage between train and val"
    assert len(val_domain & test_domain) == 0, "Leakage between val and test"
    assert len(train_domain & test_domain) == 0, "Leakage between train and test"

    # log task lengths within each split
    _train_intervals = train_domain & task_aligned_domain
    _val_intervals = val_domain & task_aligned_domain
    _test_intervals = test_domain & task_aligned_domain
    logging.info(
        f"Total train task length: {(_train_intervals.end - _train_intervals.start).sum()}"
    )
    logging.info(f"Total val task length: {(_val_intervals.end - _val_intervals.start).sum()}")
    logging.info(f"Total test task length: {(_test_intervals.end - _test_intervals.start).sum()}")

    return train_domain, val_domain, test_domain


def validate_whisker(whisker, ref_fs):
    assert np.allclose(np.diff(whisker.timestamps), 1 / ref_fs), (
        "Whisker timestamps are not at the expected fs"
    )


def validate_paws(paws, ref_fs, ref_timestamps):
    assert np.allclose(np.diff(paws.timestamps), 1 / ref_fs), (
        "Paw timestamps are not at the expected fs"
    )
    assert np.allclose(paws.timestamps, ref_timestamps), (
        "Paw timestamps do not match whisker timestamps"
    )


def validate_wheel(wheel, ref_fs, ref_timestamps, eps=1e-9):
    assert np.allclose(np.diff(wheel.timestamps), 1 / ref_fs), (
        "Wheel timestamps are not at the expected fs"
    )
    # wheel and whisker timestamps might start or end at different times,
    # Thus, we check the overlap of the inter of the two domain (with numerical instability)
    a = max(wheel.timestamps[0], ref_timestamps[0]) + eps
    b = min(wheel.timestamps[-1], ref_timestamps[-1]) + eps

    inter_wheel = wheel.timestamps[(a <= wheel.timestamps) & (wheel.timestamps <= b)]
    inter_ref = ref_timestamps[(a <= ref_timestamps) & (ref_timestamps <= b)]
    assert np.allclose(inter_wheel, inter_ref), (
        "Wheel and ref_timestamps domains do not intersect at the same times"
    )
