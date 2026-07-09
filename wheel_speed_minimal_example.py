# %% [markdown]
# # Building Neuro-AI Foundation Models with TorchBrain
#
# *A hands-on tutorial on decoding behavior with TorchBrain (Janelia NeuroDataReHack).*
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexandreAndr/janelia_ibl_bwm_talk/blob/main/wheel_speed_minimal_example.ipynb)
#
# This tutorial walks you through a minimal training pipeline for **decoding
# behavior** from spiking activity recorded in the **mouse brain with
# Neuropixels probes**, using a single session of the IBL Brain-Wide Map. Each
# session records several behavioral signals on a shared clock (wheel motion,
# whisker motion energy, and paw positions/speeds), any of which can be a
# decoding target.
#
# <!-- TODO: check how many Neuropixels probes were inserted for this session -->
#
# As a concrete, end-to-end example we decode **wheel speed** (a 1D continuous
# signal sampled at 50 Hz). Treat it as an interactive starting point: the
# **Hands On** section lets you swap in another behavioral covariate as the
# decoding target, and can be extended further using the composable transforms
# shown in the appendix.
#
# The data comes from the IBL Brain-Wide Map:
#
# [📄 Nature paper](https://www.nature.com/articles/s41586-025-09235-0){.btn .btn-outline-primary .btn-sm target="_blank"}
# [🐍 ONE API](https://int-brain-lab.github.io/ONE/){.btn .btn-outline-primary .btn-sm target="_blank"}
# [🧠 Interactive viz](https://viz.internationalbrainlab.org){.btn .btn-outline-primary .btn-sm target="_blank"}
#
# By working through this tutorial, you will learn how to:
#
# 1. Build a custom `Dataset` on top of a `brainsets` recording, so any
#    pre-processed session becomes trainable with a few lines of code.
# 2. Sample fixed-length trials around a decision-making task using
#    `TrialSampler`, the standard pattern for turning a continuous recording
#    into training examples.
# 3. Train and compare three small decoders (a linear readout, a bidirectional
#    GRU, and a dilated TCN) on the same data pipeline, so you can see how
#    architecture choice alone affects decoding performance.
#

# %% [markdown]
# # Setup
#
# Neuroscience datasets are usually distributed through a lab- or
# consortium-specific API, here the IBL's own **ONE API**. `brainsets` is a
# pipeline that wraps around that native access pattern: it downloads the raw
# session and converts it into the standardized HDF5 format that `torch-brain`
# datasets expect. The pipeline used for a given dataset can be one shared by
# [the community](https://github.com/neuro-galaxy/torch_brain/tree/main/torch_brain/pipeline/brainsets-pipelines),
# or your own local, private one, kept outside that shared collection because
# it's just an example, processes a private dataset, applies custom
# processing, or is still in development.
#
# This tutorial's own pipeline,
# [`ibl_brain_wide_map_2025`](https://github.com/AlexandreAndr/janelia_ibl_bwm_talk/tree/main/ibl_brain_wide_map_2025),
# falls in that latter category: a plain-Python `brainsets` pipeline kept
# alongside this notebook as a worked example of the real flow, for anyone
# curious how a raw IBL session gets turned into the standardized HDF5 format.
# Running it downloads/processes ~5.5 GB of
# raw data into a ~0.4 GB HDF5 file (already filtered to good-quality units,
# see "Good units only" below). For this tutorial we skip that step: the
# cells below instead fetch the single, already-processed session straight
# from the Hugging Face Hub (public, no login required), so you can get
# started in seconds. See this folder's `README.md` for more details.
#
# **Good units only.** The raw session recorded 1547 units, but many are noise
# clusters, low-firing, or sit on a probe that failed QC. Rather than filtering
# them out here at read-time, this tutorial's `brainsets` pipeline
# ([`ibl_brain_wide_map_2025/pipeline.py`](ibl_brain_wide_map_2025/pipeline.py))
# does it once, upstream, via three `--unit-filter` flags passed to
# `brainsets prepare`: `probe_qc` (`qc_neural == PASS`), `firing_rate`
# (`firing_rate > 1 Hz`), and `unit_qc` (KiloSort/IBL's own `label == 1.0`,
# "good"). Because the pipeline is just plain Python, adding this custom
# quality logic is a few lines in `extract_spikes()`, no different from any
# other preprocessing step. The processed `.h5` this notebook loads was built
# with all three flags, so it already ships with only the **358 good-quality
# units**; the discarded units' spikes were never written to disk at all.
#

# %% [markdown]
# **If running in Colab:** install this notebook's pinned dependencies
# that aren't already preinstalled.
#

# %%
# Running in Google Colab: install this notebook's pinned deps first. Colab
# starts from a blank runtime, but already has torch, scikit-learn, and
# matplotlib preinstalled, so only the packages it lacks need fetching; the
# session data itself is fetched separately from Hugging Face Hub below, so
# no repo checkout is required.
import os
import subprocess
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Keep this list in sync with requirements.txt.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch-brain==0.2.0",
            "tqdm",
            "huggingface_hub",
            "bokeh",
        ],
        check=True,
    )

# %% [markdown]
# Pick the local paths and the single session (`RECORDING_ID`) this
# tutorial decodes.

# %%
# This folder's own path, used to locate the local data dir regardless of cwd.
_HERE = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)

# Which eval session to decode; the h5 lands under DATA_ROOT/DATASET_DIRNAME.
DATA_ROOT = os.path.join(_HERE, "processed")
DATASET_DIRNAME = "ibl_brain_wide_map_2025"
RECORDING_ID = "0802ced5-33a3-405e-8336-b65ebc5cb07c"

# %% [markdown]
# Download the pre-processed session from the Hugging Face Hub, skipping
# the download if it's already on disk.

# %%
# Fetch the pre-processed session (~0.4 GB, already filtered to good-quality
# units) from the Hugging Face Hub instead of running the raw IBL download +
# brainsets pipeline (~5.5 GB raw + processing time). Skips the download if
# the file is already present locally (e.g. you
# ran `brainsets prepare` yourself). To (re)build it from scratch instead, see
# this folder's README.
_session_path = os.path.join(DATA_ROOT, DATASET_DIRNAME, f"{RECORDING_ID}.h5")
if not os.path.exists(_session_path):
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(_session_path), exist_ok=True)
    hf_hub_download(
        repo_id="AlexAndreUpenn/ibl-bwm-wheel-speed-demo",
        repo_type="dataset",
        filename=f"{RECORDING_ID}.h5",
        local_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    )

# %% [markdown]
# Imports, plus the training hyperparameters used throughout this
# tutorial.

# %%
#| code-fold: false
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

# Hyperparameters (feel free to play with these)
BIN_SIZE = 0.05  # seconds -> 20 spike bins over the 1.0 s context window
BATCH_SIZE = 64
EPOCHS = 100
# LR and BIN_SIZE below are the best of a grid over
# LR in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x BIN_SIZE in {0.01, 0.02, 0.05, 0.1},
# selected by validation R² for the TCN at 200 epochs (val 0.50, test 0.30).
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The TCN's Conv1d layers see fixed input shapes, so let cuDNN autotune kernels.
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# DataLoader performance knobs. NUM_WORKERS>0 moves each __getitem__ call
# (HDF5 read + spike binning) into separate worker processes, so it overlaps
# with the GPU doing the previous batch's forward/backward instead of
# blocking the main process. Colab's free tier usually gives 2 CPU cores, so
# we cap at that; bump this up if you have more. PIN_MEMORY (+ non_blocking=True
# on .to(device) below) lets the host->device copy run asynchronously, which
# only matters when training on a GPU.
NUM_WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = device.type == "cuda"
print(f"Using {NUM_WORKERS} dataloader workers, pin_memory={PIN_MEMORY}")

# %% [markdown]
# # A First Look at the Data
#
# Before building the dataset, let's see what a `brainsets` recording actually
# *is*. Each session is a single, lazily-loaded object that holds every modality
# on one shared time axis: the spikes of all recorded neurons, plus behavioral
# covariates (wheel, whisker, paws) and the trial structure.
#

# %% [markdown]
# We start by opening a single session with the base `Dataset` and printing it.
# What comes back is one object: some scalar metadata, plus a nested tree of typed
# containers that hold every modality on a shared clock. The full print is
# verbose, so it is tucked into a collapsible panel below; expand it to see the
# whole tree.

# %%
from torch_brain.datasets import Dataset as BaseDataset

# Open one session. This maps the HDF5 file but reads no signals yet.
peek_ds = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
)
recording = peek_ds.get_recording(RECORDING_ID)

# The full object is long, so render it inside a collapsible <details> panel
# (works in the rendered site and in Colab/Jupyter). We HTML-escape the repr
# because it contains angle brackets (e.g. "<HDF5 dataset ...>").
import html

from IPython.display import HTML, display

_repr = repr(recording)
_n_lines = _repr.count("\n") + 1
display(
    HTML(
        "<details><summary style='cursor:pointer'>"
        f"Show the full recording object (Data(...), {_n_lines} lines)"
        "</summary>"
        "<pre style='max-height:22em;overflow:auto;font-size:0.8em;line-height:1.3'>"
        f"{html.escape(_repr)}</pre></details>"
    )
)

# %% [markdown]
# ## The building blocks
#
# A recording is assembled from just five container types. Recognising them is
# enough to navigate any `brainsets` session:
#
# - **`Data`** is the container: it nests other objects plus scalar metadata. The
#   recording itself is a `Data`, and so are `session`, `subject`, `brainset`, and
#   `task_aligned_intervals`.
# - **`RegularTimeSeries`** is a signal on a fixed sampling grid (one rate, no
#   per-sample timestamps needed). Here: `wheel`, `whisker`, and `paws`, all at
#   50 Hz.
# - **`IrregularTimeSeries`** is a stream of events, each carrying its own
#   timestamp. Here: `spikes`, ~12.5 M spike times tagged with a `unit_index`.
# - **`Interval`** is a set of labelled time segments, one `start` and `end` per
#   row. Here: `trials` (424 of them), `movement_intervals`, and the `train`,
#   `val`, and `test_domain` splits.
# - **`ArrayDict`** is a table of per-item arrays sharing one axis. Here: `units`
#   (358 neurons, already filtered to good quality, each with an `id`, a 3D
#   `(x, y, z)` coordinate, a `region_cosmos` brain area, a `firing_rate`, and
#   more) and `probes`.
#
# One detail to carry into the next section: every array above printed as a
# `Lazy...` type. Nothing has been read from disk yet; the recording is still just
# a memory-mapped view of the file.

# %% [markdown]
# ## One session on a shared clock
#
# Because every modality is indexed by the same clock, we can line them up in one
# interactive figure and explore them together. The panels below share a single
# time axis (Bokeh's equivalent of matplotlib's `sharex`), so panning or zooming
# any panel moves all of them in lockstep: the spiking of all neurons (top, with
# units subsampled and their spikes thinned so a 65 min view stays responsive),
# the task trials, and three behavioral signals. This is the whole recording at a
# glance, and the raw material each training sample is later carved from.
#
# Use the toolbar (wheel zoom, box zoom, pan, reset) to drill into any stretch of
# the session. Because Bokeh draws every point in the browser, the raster and the
# 50 Hz signals are thinned for this overview; zoom in later to see a training
# window at full resolution.

# %%
#| code-fold: true
#| code-summary: "Bokeh plotting helpers (spikes, intervals, time series)"
# Small, reusable Bokeh helpers in the same style as the torch-brain tutorials:
# one for an event raster, one for a labelled-interval strip, one for a line.
# Each accepts an x_range so several panels can be locked to one shared time axis.
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import (
    BoxZoomTool,
    ColumnDataSource,
    DatetimeTickFormatter,
    FixedTicker,
    PanTool,
    Range1d,
    ResetTool,
    WheelZoomTool,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE


def show(layout, title="torch-brain tutorial figure"):
    """Render a Bokeh layout as one self-contained HTML blob (BokehJS + figure).

    `bokeh.io.show` instead relies on Jupyter "notebook comms": it loads
    BokehJS once, in whichever cell runs first, and every later `show()` call
    just assumes that copy is still on the page. That assumption breaks in
    frontends that sandbox each cell's output separately (e.g. VS Code's
    Jupyter extension), where the div silently stays empty. `file_html`
    bundles BokehJS with the figure into one blob (~4 MB, via
    `resources=INLINE`), so each cell renders on its own regardless of
    frontend, and the same HTML also drops into the rendered GitHub Pages
    site with no CDN needed at view time.
    """
    display(HTML(file_html(layout, INLINE, title)))


def _x_only_tools():
    """Fresh pan/wheel-zoom/box-zoom tools restricted to the x (time) axis.

    Each figure needs its own tool instances (Bokeh does not allow sharing a
    tool object across figures), so this is called once per panel. Restricting
    dimensions="width" keeps every interaction (drag, scroll, box) from ever
    touching a panel's y range, so the shared time axis is the only thing that
    moves or rescales.
    """
    pan = PanTool(dimensions="width")
    wheel_zoom = WheelZoomTool(dimensions="width")
    box_zoom = BoxZoomTool(dimensions="width")
    return [pan, wheel_zoom, box_zoom, ResetTool()], pan, wheel_zoom


def _time_only_formatter():
    """Fresh datetime tick formatter showing clock time only, no month/day/year.

    The x axis encodes elapsed time within a session (milliseconds since 0),
    not a calendar date, so every scale is formatted as clock time.
    """
    return DatetimeTickFormatter(
        microseconds="%H:%M:%S.%3N",
        milliseconds="%H:%M:%S.%3N",
        seconds="%H:%M:%S",
        minsec="%H:%M:%S",
        minutes="%H:%M:%S",
        hourmin="%H:%M:%S",
        hours="%H:%M:%S",
        days="%H:%M:%S",
        months="%H:%M:%S",
        years="%H:%M:%S",
    )


def _nice_step(raw_step):
    """Round a raw tick step up to a clean 1/2/5 x 10^n value."""
    if raw_step <= 0:
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    base = 10.0**exponent
    for m in (1, 2, 5, 10):
        if raw_step <= m * base:
            return m * base
    return 10.0 * base


def plot_spikes(spikes, x_range=None, width=800, height=400):
    """Raster of an event stream, from spikes.timestamps and spikes.unit_index."""
    if x_range is None:
        x_range = (spikes.timestamps[0] * 1e3, spikes.timestamps[-1] * 1e3)
    tools, pan, wheel_zoom = _x_only_tools()
    p = figure(
        x_axis_label="Time",
        y_axis_label="Unit index",
        width=width,
        height=height,
        x_axis_type="datetime",
        x_range=x_range,
        title="Spikes",
        tools=tools,
        active_drag=pan,
        active_scroll=wheel_zoom,
    )
    p.xaxis.formatter = _time_only_formatter()
    p.ygrid.grid_line_color = None
    p.yaxis.visible = False
    source = ColumnDataSource(data=dict(x=spikes.timestamps * 1e3, y=spikes.unit_index))
    p.scatter(
        "x",
        "y",
        source=source,
        size=5,
        color="navy",
        alpha=0.5,
        marker="dash",
        angle=np.pi / 2,
    )
    return p


def plot_time_series(
    data, field, index=None, x_range=None, y_axis_label=None, width=800, height=200
):
    """Line plot of one field of a time series, breaking the line over domain gaps."""
    if x_range is None:
        x_range = (data.timestamps[0] * 1e3, data.timestamps[-1] * 1e3)
    tools, pan, wheel_zoom = _x_only_tools()
    p = figure(
        x_axis_label="Time",
        y_axis_label=y_axis_label or field,
        width=width,
        height=height,
        x_axis_type="datetime",
        x_range=x_range,
        tools=tools,
        active_drag=pan,
        active_scroll=wheel_zoom,
    )
    p.xaxis.formatter = _time_only_formatter()
    p.axis.minor_tick_line_color = None
    x_values = data.timestamps * 1e3
    y_values = getattr(data, field)
    # Insert NaNs at each domain edge so the line breaks over gaps instead of
    # interpolating straight across them.
    pad = np.nan * np.ones((len(data.domain), *y_values.shape[1:]))
    x_values = np.concatenate([
        x_values,
        data.domain.start * 1e3,
        data.domain.end * 1e3,
    ])
    y_values = np.concatenate([y_values, pad, pad])
    order = np.argsort(x_values)
    x_values, y_values = x_values[order], y_values[order]
    if y_values.ndim == 2:
        assert index is not None, "index is required for a 2D field"
        y_values = y_values[:, index]
    # Fix the y range and its ticks (rather than let Bokeh auto-pick a "nice"
    # step per panel) so the wheel/whisker/paw panels all draw the same
    # number of evenly spaced horizontal gridlines, even though their data
    # ranges/units differ. BasicTicker's desired_num_ticks is only a hint and
    # still yields different counts per panel, hence the explicit FixedTicker.
    # The step is rounded to a clean 1/2/5 x 10^n value (via _nice_step)
    # instead of a raw division, so labels read e.g. "2000" not "1613.33".
    NUM_YTICKS = 4
    n_intervals = NUM_YTICKS - 1
    y_max = float(np.nanmax(y_values))
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0
    step = _nice_step(y_max / n_intervals)
    y_top = step * n_intervals
    y_ticks = [i * step for i in range(NUM_YTICKS)]
    # A small bottom pad (rather than starting the range at exactly 0) keeps
    # the trace from sitting flush on the x-axis line, matching Bokeh's usual
    # default breathing room below the data.
    p.y_range = Range1d(-0.04 * y_top, y_top)
    p.yaxis.ticker = FixedTicker(ticks=y_ticks)
    p.ygrid.ticker = FixedTicker(ticks=y_ticks)
    source = ColumnDataSource(data=dict(x=x_values, y=y_values))
    p.line(x="x", y="y", source=source, line_width=2, color="green")
    return p


def plot_intervals(*interval, x_range=None, title=None, width=800, height=200):
    """One row of rectangles per Interval passed (each rectangle is one start/end)."""
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "black",
    ]
    tools, pan, wheel_zoom = _x_only_tools()
    p = figure(
        title=title,
        x_axis_label="Time",
        x_range=x_range,
        y_axis_label="Intervals",
        y_range=(-len(interval), 1),
        width=width,
        height=height,
        x_axis_type="datetime",
        tools=tools,
        active_drag=pan,
        active_scroll=wheel_zoom,
    )
    p.xaxis.formatter = _time_only_formatter()
    p.yaxis.visible = False
    p.grid.grid_line_color = None
    for i, iv in enumerate(interval):
        centers = (iv.start + iv.end) / 2.0 * 1e3
        source = ColumnDataSource(
            data=dict(
                x=centers, width=(iv.end - iv.start) * 1e3, y=np.zeros_like(centers) - i
            )
        )
        p.rect(
            x="x",
            y="y",
            width="width",
            height=0.8,
            source=source,
            fill_color=colors[i % len(colors)],
            line_color="black",
            alpha=0.5,
        )
    return p


# %%
import gc
from types import SimpleNamespace

# A fresh handle just for this overview (all reads below are lazy).
ov_rec = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
).get_recording(RECORDING_ID)

T_END = float(ov_rec.domain.end[-1])
spk_t = np.asarray(ov_rec.spikes.timestamps)
spk_u = np.asarray(ov_rec.spikes.unit_index)
n_units = len(ov_rec.units.id)

# Bokeh draws every point in the browser (no rasterization), so for a 65 min
# overview we thin hard: keep ~70 units and cap the raster near a fixed glyph
# budget. Panning/zooming still works, and every panel moves together.
GLYPH_BUDGET = 20_000
keep = np.arange(0, n_units, max(1, n_units // 70))
m = np.isin(spk_u, keep)
sub_t = spk_t[m]
sub_row = np.searchsorted(keep, spk_u[m])  # kept unit id -> compact row index
stride = max(1, len(sub_t) // GLYPH_BUDGET)
raster = SimpleNamespace(timestamps=sub_t[::stride], unit_index=sub_row[::stride])


def _thin(obj, field, target=4000):
    """Stride a 50 Hz signal down to ~target points for a light overview line."""
    ts = np.asarray(obj.timestamps)
    y = np.asarray(getattr(obj, field))
    s = max(1, len(ts) // target)
    sh = SimpleNamespace(timestamps=ts[::s], domain=obj.domain)
    setattr(sh, field, y[::s])
    return sh


# One shared time range links every panel: Bokeh's equivalent of sharex=True.
# Times are passed in milliseconds because the helpers use a datetime x axis.
# Start zoomed in on the first slice of the session rather than the full ~65
# min recording; bounds keeps pan/zoom from ever leaving the session's extent,
# and the reset tool snaps back to this default window.
DEFAULT_ZOOM_S = 300  # 5 minutes
shared_x = Range1d(0.0, min(DEFAULT_ZOOM_S, T_END) * 1e3, bounds=(0.0, T_END * 1e3))
W = 900

p_raster = plot_spikes(raster, x_range=shared_x, width=W, height=360)
p_raster.title.text = (
    f"One recording, one shared clock ({len(keep)} of {n_units} neurons)"
)

p_trials = plot_intervals(
    ov_rec.trials, x_range=shared_x, title="task trials", width=W, height=80
)

p_wheel = plot_time_series(
    _thin(ov_rec.wheel, "speed"),
    "speed",
    x_range=shared_x,
    y_axis_label="wheel speed",
    width=W,
    height=130,
)
p_whisk = plot_time_series(
    _thin(ov_rec.whisker, "motion_energy"),
    "motion_energy",
    x_range=shared_x,
    y_axis_label="whisker ME",
    width=W,
    height=130,
)
p_paw = plot_time_series(
    _thin(ov_rec.paws, "left_paw_speed"),
    "left_paw_speed",
    x_range=shared_x,
    y_axis_label="L paw speed",
    width=W,
    height=130,
)

# Only the bottom panel needs to show the (shared) time axis.
for p in (p_raster, p_trials, p_wheel, p_whisk):
    p.xaxis.visible = False
p_paw.xaxis.axis_label = "time in session"

show(column(p_raster, p_trials, p_wheel, p_whisk, p_paw))

# Free the spike arrays we pulled in just for the plot.
del spk_t, spk_u, sub_t, sub_row, raster, ov_rec
_ = gc.collect()

# %% [markdown]
# ## Lazy loading: pay only for what you touch
#
# This session is **~0.43 GB** on disk. Reading all of it into memory at once is
# both possible and a bad idea: `materialize()` pulls every array into RAM, and
# the footprint below would then be paid again for every session a
# foundation model trains on.

# %%
import time


def _rss_gb():
    """Resident memory of this process, in GB (Linux; None elsewhere)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6  # kB -> GB
    except OSError:
        return None


fpath = os.path.join(DATA_ROOT, DATASET_DIRNAME, f"{RECORDING_ID}.h5")
print(f"session on disk:              {os.path.getsize(fpath) / 1e9:5.2f} GB")

eager_rec = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
).get_recording(RECORDING_ID)

before = _rss_gb()
t = time.time()
full = eager_rec.materialize()  # read EVERYTHING into memory (what we avoid)
dt_materialize = time.time() - t
after = _rss_gb()
if before is not None:
    print(
        f"materialize() into RAM:       {dt_materialize:5.2f} s   "
        f"(+{after - before:4.2f} GB resident)"
    )
else:
    print(f"materialize() into RAM:       {dt_materialize:5.2f} s")

# Drop it again so we do not carry the materialized session through the rest
# of the notebook.
del full, eager_rec
_ = gc.collect()

# %% [markdown]
# But a training step never needs the whole session. It reads a short **temporal
# slice** and, within it, only the **input** and the **target** (here the spikes
# and one behavioral signal). Everything else, every other modality and every
# other time point, can stay on disk. That is what lazy loading buys: the
# recording is memory-mapped, and you pay (bytes read, RAM used) only for the
# attributes and time windows you actually access.
#
# Two independent axes of savings, each measured on a fresh, lazily-opened
# recording (reads no signals until accessed), against the `materialize()`
# runtime above:

# %%
# (i) Slice to a single 1.0 s window, THEN materialize it: every modality is
#     still read, but only for that one second, not the full ~65 min session.
lazy_rec = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
).get_recording(RECORDING_ID)

mv = lazy_rec.task_aligned_intervals.movement_intervals
t0 = float(np.asarray(mv.start)[10])

t = time.time()
materialized_window = lazy_rec.slice(t0, t0 + 1.0).materialize()
dt_slice = time.time() - t
print(
    f"(i) slice 1.0 s window, then materialize(): {dt_slice * 1e3:6.2f} ms\n"
    f"    -> {dt_materialize / dt_slice:6.0f}x less time than materializing the whole session"
)

# (ii) Access a single attribute (wheel speed) for the whole session: every
#      time point is read, but only for that one modality, not the others.
lazy_rec = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
).get_recording(RECORDING_ID)

t = time.time()
wheel_speed = np.asarray(lazy_rec.wheel.speed)
dt_attr = time.time() - t
print(
    f"(ii) load wheel.speed only, whole session:  {dt_attr * 1e3:6.2f} ms\n"
    f"    -> {dt_materialize / dt_attr:6.0f}x less time than materializing the whole session"
)

# %% [markdown]
# ::: {.callout-tip}
# # Why this matters for neuro foundation models
#
# - **Fast, minimal reads.** Each training step loads only the variables and the
#   short time window the model needs, so a step touches megabytes, not gigabytes,
#   no matter how large the underlying session is.
# - **Store more at no model cost.** The file can keep everything (all units, every
#   behavioral signal, raw QC fields, alternative labels) for future benchmarking
#   or new tasks. Unused fields cost disk space, not RAM or training time, because
#   they are simply never read.
# - **Defer processing to the slice.** Since each window is tiny, per-sample steps
#   (binning, normalization, adding noise) are cheap to compute on the fly. You can
#   reparametrize them, or try variants, without ever reprocessing the file on disk.
# :::

# %% [markdown]
# ## Transforms: deferring computation to each slice
#
# That last point is exactly what a **transform** is: a small function applied to a
# window right after the lazy slice and before the model, inside `__getitem__`.
# Because it runs per sample on a tiny crop, it is cheap, and swapping or chaining
# transforms reshapes the pipeline without touching the data on disk or the model.
#
# Binning is the clearest example. Spikes arrive as an irregular event stream, but
# the model wants a regular grid, so each window is binned on the fly. The bin
# width is just a parameter: change it and the model input reshapes, with no
# reprocessing.

# %%
from torch_brain.utils import bin_spikes

window = lazy_rec.slice(t0, t0 + 1.0)
for bs in (0.02, 0.05, 0.10):
    Xb = bin_spikes(window.spikes, num_units=len(window.units), bin_size=bs)
    print(f"bin_size = {bs:.2f} s  ->  X shape {tuple(Xb.shape)}  (bins x units)")

# %% [markdown]
# The same idea extends to context length, augmentation, unit selection, and
# sampling jitter: each is a one-line change to the pipeline. The appendix,
# *Why this framework is powerful*, works through them.

# %% [markdown]
# # Dataset, Sampler, and DataLoader
#
# TorchBrain's data pipeline for training is built from three cooperating
# pieces:
#
# - **Dataset** tells the sampler *where* sampling is allowed (via
#   `get_sampling_intervals`), and turns a chosen window into an `(X, y)`
#   sample (via `__getitem__`).
# - **Sampler** decides *what* samples to load, and in what order, by
#   emitting `DatasetIndex` objects.
# - **DataLoader** fetches the chosen samples and collates them into a batch,
#   as usual in PyTorch.

# %% [markdown]
# ## Defining a custom Dataset
#
# In TorchBrain we define a custom `Dataset`; `IBLBrainWideMap2025` below is a
# simple example, for a single recording. Two methods matter for the
# wheel-speed task:
#
# - **`get_sampling_intervals`**: decides *which* time windows count as
#   samples. For wheel-speed decoding, each sample is a fixed **1.0 s** window
#   drawn from the trials of the IBL decision-making task, restricted to the
#   movement window and to the times where the wheel signal is defined. We use
#   the dataset's built-in **causal** train/val/test split (`{split}_domain`):
#   train is early in the session, val is in the middle, and test is late
#   (more on the actual split proportions below).
# - **`__getitem__`**: given a time window, turns it into an `(X, y)` sample:
#   `X` is the model input, a binned spike raster; `y` is the model target,
#   the wheel speed.
#
# %%
#| code-fold: false
from pathlib import Path
from typing import Literal

from torch_brain.datasets import Dataset, DatasetIndex, SpikingDatasetMixin
from torch_brain.utils import bin_spikes


class IBLBrainWideMap2025(SpikingDatasetMixin, Dataset):
    """
    Dataset for the IBL Brain-Wide Map (2025), for a single recording.

    Args:
        root: The root directory of the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        recording_id: The recording id to load.
        bin_size: The spike-binning width in seconds. Only required when the
            dataset is actually indexed (i.e. not for exploration-only use).
        split: The split of the dataset (train, val, test), or None to skip
            split-based interval filtering (e.g. when just exploring a recording).

    """

    # wheel speed is a 1D continuous signal, regularly sampled at BEHAVIOR_SFREQ (50 Hz).
    out_dim = 1
    spiking_dataset_mixin_uniquify_unit_ids = True
    CONTEXT_WINDOW = 1.0  # seconds
    BEHAVIOR_SFREQ = 50  # Hz

    def __init__(
        self,
        root: str,
        recording_id: str,
        bin_size: float,
        dirname: str = "ibl_brain_wide_map_2025",
        split: Literal["train", "val", "test"] | None = "train",
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=[recording_id],
        )

        # store some attributes that are useful later
        self.split = split
        self.recording_id = recording_id
        self.bin_size = bin_size
        self.out_sampling_rate = float(self.BEHAVIOR_SFREQ)  # 50 Hz
        self.out_samples = round(self.CONTEXT_WINDOW * self.out_sampling_rate)  # 50
        self.num_bins = round(self.CONTEXT_WINDOW / self.bin_size)
        self.num_units = len(self.get_unit_ids())

    # get_sampling_intervals() returns {recording_id: Interval} listing
    # the windows the sampler may draw from.
    def get_sampling_intervals(self, *_args, **_kwargs):
        rid = self.recording_id
        recording = self.get_recording(rid)

        intervals = recording.task_aligned_intervals.movement_intervals
        intervals = intervals & getattr(recording, f"{self.split}_domain")
        intervals = intervals & recording.wheel._domain
        return {rid: intervals}

    # `index` is a DatasetIndex(recording_id, start, end) produced by the sampler.
    def __getitem__(self, index: DatasetIndex):
        recording = self.get_recording(index.recording_id)
        data = recording.slice(index.start, index.end)

        # All models take (num_bins, num_units) and return (out_samples, out_dim).

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()  # shape: (num_bins, num_units)

        Y = np.asarray(data.wheel.speed, dtype=np.float32)  # shape: (out_samples,)
        Y = torch.from_numpy(Y).unsqueeze(-1)  # shape: (out_samples, out_dim=1)

        return X, Y


# %% [markdown]
# ## Building the train, validation, and test pipeline
#
# `IBLBrainWideMap2025` is instantiated once per split, each wrapped in its
# own `TrialSampler` and `DataLoader`. Once these are built, we will look at
# what the split boundaries and a single sample actually look like, and step
# through a real training epoch's sample order interactively.

# %%
from torch.utils.data import DataLoader  # standard PyTorch loader
from torch_brain.samplers import TrialSampler

train_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="train", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
# We want to sample "one-trial-at-a-time", so we use the TrialSampler
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(),
    shuffle=True,
)
# Note the sampler is passed explicitly; it is not the default random/sequential
# sampler PyTorch picks for an indexable dataset.
# num_workers>0 loads batches in separate worker processes (overlapping with
# GPU compute instead of blocking on it); persistent_workers keeps them alive
# across epochs instead of respawning; pin_memory lets .to(device) below copy
# asynchronously.
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    persistent_workers=NUM_WORKERS > 0,
    pin_memory=PIN_MEMORY,
)
print(f"Number of units: {train_ds.num_units}")
print(f"Number of training samples: {len(train_sampler)}")

# Validation Dataset, Sampler, and DataLoader
val_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="val", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
val_sampler = TrialSampler(sampling_intervals=val_ds.get_sampling_intervals())
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    persistent_workers=NUM_WORKERS > 0,
    pin_memory=PIN_MEMORY,
)
print(f"Number of validation samples: {len(val_sampler)}")

# Test Dataset, Sampler, and DataLoader.
# The test split is the *late* part of the session (causal split). We touch it
# only once, for the final score after training/model selection is done.
test_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="test", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
test_sampler = TrialSampler(sampling_intervals=test_ds.get_sampling_intervals())
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    sampler=test_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
print(f"Number of test samples: {len(test_sampler)}")

print(f"Number of units:  {train_ds.num_units}")
print(f"Bins per sample:  {train_ds.num_bins}  (bin size = {BIN_SIZE}s)")
print(f"Target samples:   {train_ds.out_samples}  (at {train_ds.out_sampling_rate} Hz)")
print(
    f"Train trials:     {len(train_ds.get_sampling_intervals()[train_ds.recording_id])}"
)
print(f"Val trials:       {len(val_ds.get_sampling_intervals()[val_ds.recording_id])}")

# %% [markdown]
# Let's first peek at a single sample to confirm the shapes match what we expect.

# %%
first_sample_index = next(iter(train_sampler))
print(
    f"First sample:\n"
    f"    recording_id: {first_sample_index.recording_id},\n"
    f"    start time: {first_sample_index.start},\n"
    f"    end time: {first_sample_index.end}\n"
)

X, Y = train_ds[first_sample_index]
print(f"X shape: {tuple(X.shape)}  (num_bins, num_units)")
print(f"Y shape: {tuple(Y.shape)}  (out_samples, out_dim)")

# %% [markdown]
# ## Visualizing the split, a sample, and how sampling works
#
# `{split}_domain` is one contiguous block per split (train early, val in the
# middle, test late), covering the whole session. But `TrialSampler` never
# samples from that block directly: `get_sampling_intervals` first intersects
# it with `movement_intervals` (only movement periods count as samples) and
# with `wheel._domain` (only where the target signal is defined). The first
# two rows below make that concrete: the domain blocks, and directly beneath
# them, the much sparser set of movement windows each split actually draws
# samples from.
#
# One thing this reveals: the split is balanced by **trial count**, close to
# the pipeline's target 40/20/40 (148/72/145 movement windows), but because
# trials are not paced evenly through the session, the same split covers only
# about **21%/11%/67% of the session's raw time**. Pan or zoom either row
# (they are linked) to see this for yourself.
#
# `train_sampler` was built with `shuffle=True`, so every epoch
# `TrialSampler.__iter__` draws a fresh `torch.randperm` and yields those
# windows in a new random order; `val_sampler`/`test_sampler` stay in the
# fixed order `sampling_intervals` was built in, for reproducible evaluation.
# The slider below freezes one such training-epoch order: step through it to
# see the corresponding window highlighted on the row above, and the
# `(X, Y)` pair, binned spikes and wheel speed, it turns into.

# %%
#| code-fold: true
#| code-summary: "Bokeh: split domains, sampling intervals, and an interactive sample stepper"
from bokeh.layouts import row as bokeh_row
from bokeh.models import BoxAnnotation, Button, CustomJS, Div, Slider

recording = train_ds.get_recording(RECORDING_ID)
split_t_end = float(recording.domain.end[-1])
split_x_range = Range1d(0.0, split_t_end * 1e3, bounds=(0.0, split_t_end * 1e3))

domains_fig = plot_intervals(
    recording.train_domain,
    recording.val_domain,
    recording.test_domain,
    x_range=split_x_range,
    title="train / val / test domain: one contiguous block each",
    width=900,
    height=90,
)
domains_fig.xaxis.visible = False

train_intervals = train_ds.get_sampling_intervals()[RECORDING_ID]
val_intervals = val_ds.get_sampling_intervals()[RECORDING_ID]
test_intervals = test_ds.get_sampling_intervals()[RECORDING_ID]

samples_fig = plot_intervals(
    train_intervals,
    val_intervals,
    test_intervals,
    x_range=split_x_range,
    title="sampling intervals actually drawn: movement windows within each split",
    width=900,
    height=90,
)
samples_fig.xaxis.axis_label = "time in session"

legend = Div(
    text=(
        "<span style='color:red'>&#9632;</span> train &nbsp;&nbsp;"
        "<span style='color:blue'>&#9632;</span> val &nbsp;&nbsp;"
        "<span style='color:green'>&#9632;</span> test"
    )
)

# The moving highlight: one training window at a time, drawn on the row above.
highlight = BoxAnnotation(
    left=0, right=0, bottom=-0.45, top=0.45, fill_color="red", fill_alpha=0.35
)
samples_fig.add_layout(highlight)

# Freeze one shuffled training-epoch order and precompute its (X, Y) pairs so
# the slider below only ever swaps already-computed arrays, no HDF5 reads.
N_DEMO = 12
demo_indices = list(train_sampler)[:N_DEMO]
demo_X = []
demo_Y = []
for idx in demo_indices:
    Xd, Yd = train_ds[idx]
    demo_X.append(Xd.T.numpy().tolist())  # (num_units, num_bins), for the image glyph
    demo_Y.append(Yd[:, 0].numpy().tolist())  # (out_samples,)
demo_starts = [float(idx.start) for idx in demo_indices]
demo_ends = [float(idx.end) for idx in demo_indices]
t_local = np.linspace(0.0, train_ds.CONTEXT_WINDOW, train_ds.out_samples).tolist()

raster_source = ColumnDataSource(data=dict(image=[demo_X[0]]))
p_demo_raster = figure(
    width=380,
    height=280,
    title="Binned spikes (X)",
    x_axis_label="Time bin",
    y_axis_label="Unit",
    toolbar_location=None,
)
p_demo_raster.image(
    image="image",
    x=0,
    y=0,
    dw=train_ds.num_bins,
    dh=train_ds.num_units,
    source=raster_source,
    palette="Greys256",
)

wheel_source = ColumnDataSource(data=dict(x=t_local, y=demo_Y[0]))
p_demo_wheel = figure(
    width=380,
    height=280,
    title="Wheel speed (Y)",
    x_axis_label="Time within window (s)",
    y_axis_label="Wheel speed",
    toolbar_location=None,
)
p_demo_wheel.line("x", "y", source=wheel_source, line_width=2, color="green")

info_div = Div(
    text=f"Sample 1 / {N_DEMO} (start={demo_starts[0]:.2f}s, end={demo_ends[0]:.2f}s)"
)
slider = Slider(
    start=0,
    end=N_DEMO - 1,
    value=0,
    step=1,
    title="Sample index, in the order this epoch drew them",
)
prev_btn = Button(label="< Prev", width=70)
next_btn = Button(label="Next >", width=70)

step_callback = CustomJS(
    args=dict(
        slider=slider,
        raster_source=raster_source,
        wheel_source=wheel_source,
        info_div=info_div,
        highlight=highlight,
        X_list=demo_X,
        Y_list=demo_Y,
        starts=demo_starts,
        ends=demo_ends,
        t_local=t_local,
        n_demo=N_DEMO,
    ),
    code="""
    const i = slider.value
    raster_source.data = {image: [X_list[i]]}
    wheel_source.data = {x: t_local, y: Y_list[i]}
    highlight.left = starts[i] * 1e3
    highlight.right = ends[i] * 1e3
    info_div.text = `Sample ${i + 1} / ${n_demo} (start=${starts[i].toFixed(2)}s, end=${ends[i].toFixed(2)}s)`
    """,
)
slider.js_on_change("value", step_callback)
prev_btn.js_on_click(
    CustomJS(
        args=dict(slider=slider), code="slider.value = Math.max(0, slider.value - 1)"
    )
)
next_btn.js_on_click(
    CustomJS(
        args=dict(slider=slider, n_demo=N_DEMO),
        code="slider.value = Math.min(n_demo - 1, slider.value + 1)",
    )
)
# Set the initial highlight position (the callback above only fires on change).
highlight.left = demo_starts[0] * 1e3
highlight.right = demo_ends[0] * 1e3

show(
    column(
        legend,
        domains_fig,
        samples_fig,
        bokeh_row(prev_btn, slider, next_btn),
        info_div,
        bokeh_row(p_demo_raster, p_demo_wheel),
    )
)

# %% [markdown]
# # The Model
#
# Three small decoders are defined in the cells below: Linear, GRU, and TCN.
#
# - **Linear**: flatten + a single `nn.Linear` layer.
# - **GRU**: bidirectional GRU, then a per-timestep linear readout and an
#   interpolation to upsample from `num_bins` to `out_samples`.
# - **TCN**: a stack of dilated 1D convolutions, followed by the same
#   interpolation + readout.
#
# All three follow the same interface. They take `(batch, num_bins, num_units)`
# and return `(batch, out_samples, out_dim)`.

# %% [markdown]
# ## Linear


# %%
class Linear(nn.Module):
    def __init__(self, in_units, in_bins, out_dim, out_samples, dropout=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.out_samples = out_samples

        input_size = in_units * in_bins
        output_size = out_dim * out_samples
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_size, output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        y = self.net(x.flatten(start_dim=1))
        y = y.view(batch_size, self.out_samples, self.out_dim)
        return y


# %% [markdown]
# ## GRU


# %%
class GRU(nn.Module):
    def __init__(
        self,
        in_units,
        in_bins,
        out_dim,
        out_samples,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.out_samples = out_samples

        self.gru = nn.GRU(
            input_size=in_units,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.readout = nn.Linear(
            in_features=2 * hidden_dim if bidirectional else hidden_dim,
            out_features=out_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        z, _ = self.gru(x)
        y = self.readout(z)
        y = y.permute(0, 2, 1)  # (B, T, D) ->  (B, D, T)
        y = nn.functional.interpolate(y, self.out_samples, mode="linear")
        y = y.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return y


# %% [markdown]
# ## TCN


# %%
class TCN(nn.Module):
    def __init__(
        self,
        in_units,
        in_bins,
        out_dim,
        out_samples,
        hidden_dim=64,
        num_layers=8,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.out_samples = out_samples

        layers = []
        in_channels = in_units
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = hidden_dim
        self.net = nn.Sequential(*layers)
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        z = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        z = self.net(z)
        z = nn.functional.interpolate(z, self.out_samples, mode="linear")
        z = z.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        y = self.readout(z)
        return y


# %% [markdown]
# ## Instantiating the model

# %%
model = TCN(  # try: Linear, GRU, TCN
    in_units=train_ds.num_units,
    in_bins=train_ds.num_bins,
    out_dim=train_ds.out_dim,
    out_samples=train_ds.out_samples,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {num_params:,}")
print(model)

# %% [markdown]
# # Training and Evaluation
#
# With the data pipeline and model defined, we train, then evaluate on held-out
# data: first the validation set (monitored during training), then the test
# set, touched only once, after training and model selection are done.

# %% [markdown]
# ## Training
#
# A standard PyTorch loop! MSE loss against the wheel speed, AdamW optimizer,
# R² score on the validation set at the end of each epoch.

# %%
from sklearn.metrics import r2_score

optim = torch.optim.AdamW(model.parameters(), lr=LR)

val_r2_history = []

for _epoch in (epoch_pbar := tqdm(range(EPOCHS))):
    model.train()
    for X, Y in train_loader:
        # non_blocking=True only actually overlaps with pin_memory=True (set
        # on the DataLoader above) and a CUDA device; it is a no-op otherwise.
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        model.eval()
        preds, targets = [], []
        for X, Y in val_loader:
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            preds.append(model(X))
            targets.append(Y)
        pred = torch.cat(preds).flatten(0, 1).cpu()
        target = torch.cat(targets).flatten(0, 1).cpu()
        r2 = r2_score(target, pred)
        val_r2_history.append(r2)
        epoch_pbar.set_description(f"Val R²: {r2:.3f}")

# %% [markdown]
# ## Evaluation
#
# Plot the R² curve over training and compare predicted vs. actual wheel speed
# on several validation trials.

# %%
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(val_r2_history)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation R²")
ax.set_title("Validation R² over training")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's look at examples of how our model's predictions compare with the ground
# truth, across 8 validation trials.

# %%
N_EXAMPLES = 8
val_indices = list(val_sampler)[:N_EXAMPLES]
t = np.linspace(0.0, val_ds.CONTEXT_WINDOW, val_ds.out_samples)

model.eval()
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
with torch.no_grad():
    for ax, index in zip(axes.flat, val_indices):
        X, Y = val_ds[index]
        pred = model(X.unsqueeze(0).to(device)).squeeze(0).cpu()
        ax.plot(t, Y[:, 0].numpy(), label="actual", color="k")
        ax.plot(t, pred[:, 0].numpy(), label="predicted", color="green")
        ax.set_title(f"start={index.start:.2f}s", fontsize=9)

for ax in axes[-1, :]:
    ax.set_xlabel("Time within trial (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Wheel speed")
axes[0, 0].legend(loc="upper left")
fig.suptitle("Predicted vs. actual wheel speed (8 validation trials)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final Test Evaluation
#
# The **test** split is the *late* portion of the session (the causal split), held
# out from training and model selection. We score it exactly **once**, here, to get
# an honest estimate of decoding performance. Because the split is temporal rather
# than shuffled, test R² is typically **lower** than validation R² -- that gap
# reflects within-session nonstationarity, which is exactly what a causal split is
# designed to expose.

# %%
model.eval()
with torch.no_grad():
    preds, targets = [], []
    for X, Y in test_loader:
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        preds.append(model(X))
        targets.append(Y)
    test_pred = torch.cat(preds).flatten(0, 1).cpu()
    test_target = torch.cat(targets).flatten(0, 1).cpu()

test_r2 = r2_score(test_target, test_pred)
best_val_r2 = max(val_r2_history)
print(f"Best validation R²: {best_val_r2:.3f}")
print(f"Final test R²:      {test_r2:.3f}")

# %% [markdown]
# And the predicted vs. actual wheel speed across 8 held-out test trials:

# %%
N_EXAMPLES = 8
test_indices = list(test_sampler)[:N_EXAMPLES]
t = np.linspace(0.0, test_ds.CONTEXT_WINDOW, test_ds.out_samples)

model.eval()
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
with torch.no_grad():
    for ax, index in zip(axes.flat, test_indices):
        X, Y = test_ds[index]
        pred = model(X.unsqueeze(0).to(device)).squeeze(0).cpu()
        ax.plot(t, Y[:, 0].numpy(), label="actual", color="k")
        ax.plot(t, pred[:, 0].numpy(), label="predicted", color="green")
        ax.set_title(f"start={index.start:.2f}s", fontsize=9)

for ax in axes[-1, :]:
    ax.set_xlabel("Time within trial (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Wheel speed")
axes[0, 0].legend(loc="upper left")
fig.suptitle("Test trials: predicted vs. actual wheel speed (8 trials)")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Hands On: Decode a Different Covariate
#
# Now it's your turn. Everything you need is already defined above: the dataset
# class, the samplers, and the training loop. This exercise focuses on the one
# piece that changes when you decode a new signal: **what `__getitem__` returns
# as `Y`**.
#
# **Goal:** instead of wheel speed, decode one of the other covariates recorded
# in this session: `whisker.motion_energy`, `paws.left_paw_speed`, or
# `paws.right_paw_speed`.
#
# **Steps:**
# 1. Pick a covariate from the list above.
# 2. Subclass `IBLBrainWideMap2025` (skeleton below) and point `Y` at your
#    chosen covariate instead of `data.wheel.speed`.
# 3. Instantiate train/val/test datasets, samplers, and loaders for your new
#    class, exactly as in "Creating the Datasets, Samplers, and DataLoaders".
# 4. Re-run the training loop on a fresh model instance, and compare its
#    validation R² to wheel speed's.
#
# This is a self-check: if your new covariate trains and its R² is in a
# plausible range, your subclass is wired correctly. If you want to go further,
# this is also the natural place to try the transforms from the appendix below
# (a finer bin size, a longer context window, unit dropout, ...) on your new
# target.
#


# %%
class IBLCovariateDataset(IBLBrainWideMap2025):
    """Same as IBLBrainWideMap2025, but decodes `namespace.attr` instead of wheel speed."""

    def __init__(self, *args, namespace: str, attr: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = namespace
        self.attr = attr

    def __getitem__(self, index):
        recording = self.get_recording(index.recording_id)
        data = recording.slice(index.start, index.end)

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()

        # TODO: read your chosen covariate instead of the wheel speed, e.g.
        # Y = np.asarray(getattr(getattr(data, self.namespace), self.attr), dtype=np.float32)
        Y = np.asarray(data.wheel.speed, dtype=np.float32)  # <-- replace this line
        Y = torch.from_numpy(Y).unsqueeze(-1)

        return X, Y


# TODO: pick a covariate to decode
NAMESPACE = "whisker"  # try: "whisker", "paws"
ATTR = "motion_energy"  # try: "motion_energy", "left_paw_speed", "right_paw_speed"

# TODO: build train/val/test IBLCovariateDataset instances (passing
# namespace=NAMESPACE, attr=ATTR), wrap each in a TrialSampler + DataLoader
# (same pattern as "Creating the Datasets, Samplers, and DataLoaders" above),
# then re-run the training loop on a fresh model instance and compare its
# validation R² to wheel speed's.


# %% [markdown]
# ---
# # Appendix: Why this framework is powerful
#
# The training pipeline above is four independent pieces: **Dataset** (*where* to
# sample), **Sampler** (*which* windows), **Transform** (*how* to process), and
# **DataLoader** (*how* to batch). Because they are decoupled, common neuroscience
# experiments (finer binning, more context, augmentation, region ablations,
# jittered sampling) are each a **one-line change that never touches the model or
# the training loop**.
#
# Every cell below is illustrative: it operates on a single demo window and does
# **not** modify the tuned pipeline above.

# %% [markdown]
# # 1. Lazy loading and the time gain
#
# A `brainsets` recording is memory-mapped: `data.slice(start, end)` reads only
# the bytes for the window you ask for. You open a session in milliseconds
# regardless of its size, and each sample touches a tiny slice of it, so you
# never hold the session (or even one full modality) in RAM. That is what
# makes training over a 65-min session, or scaling to many sessions, feasible
# at all.
#
# The catch: lazy is cheap *per call*, but `IBLBrainWideMap2025.__getitem__`
# redoes the slice + bin on *every* epoch, since it does not cache. For this
# tutorial's scale (a single session, ~1 ms per window) that is a deliberate
# trade: a few extra seconds of redundant HDF5 reads and binning over a full
# training run, in exchange for a `__getitem__` with no memoization state to
# reason about. Memoizing the handful of windows a run actually samples (keyed
# on `(recording_id, start, end)`) would trade that simplicity back for speed
# if this were scaled to more sessions, more epochs, or heavier per-sample
# processing.
#
# # %%
# import os
# import time
#
# fpath = os.path.join(DATA_ROOT, "ibl_brain_wide_map_2025", f"{RECORDING_ID}.h5")
# print(f"session file on disk:       {os.path.getsize(fpath) / 1e9:5.2f} GB")
#
# t = time.time()
# _d = IBLBrainWideMap2025(root=DATA_ROOT, recording_id=RECORDING_ID, split=None)
# _r = _d.get_recording(RECORDING_ID)
# t_open = time.time() - t
#
# t = time.time()
# for _ in range(20):
#     _s = _r.slice(DEMO_T0, DEMO_T0 + 1.0)
#     _ = bin_spikes(_s.spikes, num_units=len(_s.units), bin_size=BIN_SIZE)
# t_slice = (time.time() - t) / 20
#
# t = time.time()
# _allt = np.asarray(_r.spikes.timestamps)
# t_all = time.time() - t
#
# print(f"open dataset (lazy):        {t_open * 1000:5.1f} ms   (reads no signals)")
# print(
#     f"one 1.0 s window slice+bin: {t_slice * 1000:5.1f} ms   (reads only that window)"
# )
# _all_mb = _allt.nbytes / 1e6
# print(
#     f"load ALL spike timestamps:  {t_all * 1000:5.0f} ms   ({_all_mb:.0f} MB, one modality)"
# )
#
# # %% [markdown]
# # # 2. Change the bin size
# #
# # Binning is one parameter. Finer bins give more temporal resolution but a
# # higher-dimensional, sparser input; coarser bins regularize. The model still
# # just receives `(num_bins, num_units)`.
#
# # %%
# fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
# for ax, bs in zip(axes, [0.02, 0.05, 0.1]):
#     s = demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)
#     Xd = bin_spikes(s.spikes, num_units=len(s.units), bin_size=bs)
#     ax.imshow(
#         np.log1p(Xd.T),
#         aspect="auto",
#         cmap="Greys",
#         origin="lower",
#         interpolation="nearest",
#     )
#     ax.set_title(f"bin_size = {bs} s  ({Xd.shape[0]} bins)")
#     ax.set_xlabel("time bin")
# axes[0].set_ylabel("unit")
# fig.suptitle("Same 1.0 s window, three bin sizes: one parameter, no model change")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 3. Longer or shorter context window
# #
# # The context window is how much time each sample spans; it lives in the
# # sampler/slice length, not in the data. A 2.0 s window simply yields 100 target
# # samples (at 50 Hz) and `2.0 / bin_size` input bins.
#
# # %%
# fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for ax, L in zip(axes, [0.5, 1.0, 2.0]):
#     s = demo_rec.slice(DEMO_T0, DEMO_T0 + L)
#     y = np.asarray(s.wheel.speed)
#     ax.plot(np.linspace(0, L, len(y)), y, color="#4C72B0")
#     ax.set_title(f"{L} s -> {len(y)} target samples")
#     ax.set_xlabel("time in window (s)")
# axes[0].set_ylabel("wheel speed")
# fig.suptitle("Same start, three context windows: only the slice length changes")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 4. Augmentation via composable transforms
# #
# # Transforms attach with `transform=` and chain with `Compose`; they run inside
# # `__getitem__`, so augmentation is re-drawn every epoch. Here a random 0.8 s crop
# # of the same window (a different crop each call). *(Time-warping augmentations
# # like `RandomTimeScaling` also exist; we revisit those separately.)*
#
# # %%
# augment = Compose([RandomCrop(0.8)])
# base = demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)
# y_base = np.asarray(base.wheel.speed)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
# axes[0].plot(np.linspace(0, 1.0, len(y_base)), y_base, color="#333333")
# axes[0].set_title(f"raw (1.0 s, {len(y_base)} samples)")
# for _ in range(3):
#     a = augment(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0))
#     y_a = np.asarray(a.wheel.speed)
#     axes[1].plot(np.linspace(0, len(y_a) / 50.0, len(y_a)), y_a, alpha=0.7)
# axes[1].set_title("RandomCrop(0.8): a fresh crop each epoch")
# for ax in axes:
#     ax.set_xlabel("time (s)")
# axes[0].set_ylabel("wheel speed")
# fig.suptitle("A transform re-draws every epoch, combating overfitting for free")
# plt.tight_layout()
# plt.show()
#
# # %% [markdown]
# # # 5. Mask or select units
# #
# # Units are a labelled axis, so you can drop or select them declaratively.
# # `UnitDropout` keeps a random subset each sample (augmentation / regularizer);
# # `UnitFilter` keeps a fixed subset, e.g. one brain region, turning "which region
# # drives wheel decoding?" into a one-line ablation.
#
# # %%
# # (a) UnitDropout: a different random subset of units each call
# for _ in range(5):
#     before = len(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0).units)
#     after = len(
#         UnitDropout(field="spikes")(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)).units
#     )
#     print(f"UnitDropout: {before} -> {after} units kept")
#
# # (b) UnitFilter: keep only units in one brain region (cosmos atlas)
# regions = np.asarray(demo_rec.units.region_cosmos)
# labels, counts = np.unique(regions, return_counts=True)
# fig, ax = plt.subplots(figsize=(7, 3))
# ax.bar([r.decode() for r in labels], counts, color="#4C72B0")
# ax.set_ylabel("units")
# ax.set_title("Units per brain region: UnitFilter selects any subset in one line")
# plt.tight_layout()
# plt.show()
#
# th_filter = UnitFilter(
#     mask_fn=lambda u: np.asarray(u.region_cosmos) == b"TH", field="spikes"
# )
# n_before = len(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0).units)
# n_after = len(th_filter(demo_rec.slice(DEMO_T0, DEMO_T0 + 1.0)).units)
# print(f"UnitFilter to thalamus: {n_before} -> {n_after} units")
#
# # %% [markdown]
# # # 6. Sampling window with jitter
# #
# # `TrialSampler` emits the same fixed windows every epoch (reproducible eval).
# # `RandomFixedWindowSampler` applies a fresh random temporal jitter each epoch, so
# # the model sees slightly shifted positions within every interval: more effective
# # data, less positional overfitting. Only the sampler changes.
#
# # %%
# jitter_sampler = RandomFixedWindowSampler(
#     sampling_intervals={RECORDING_ID: demo_iv}, window_length=1.0
# )
# fixed_sampler = TrialSampler(sampling_intervals={RECORDING_ID: demo_iv})
#
#
# def first_starts(sampler, n=6):
#     return [round(float(idx.start), 2) for idx in list(sampler)[:n]]
#
#
# print("RandomFixedWindowSampler (jitter, new positions each epoch):")
# print("  epoch 1:", first_starts(jitter_sampler))
# print("  epoch 2:", first_starts(jitter_sampler))
# print("TrialSampler (fixed windows, identical each epoch):")
# print("  epoch 1:", first_starts(fixed_sampler))
# print("  epoch 2:", first_starts(fixed_sampler))
