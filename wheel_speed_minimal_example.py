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
            "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@deb39f026da33a93f9a95884eda82c1aa60dcd1a",
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
#| code-fold: show
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

# Hyperparameters (feel free to play with these)
BIN_SIZE = 0.05  # seconds -> 20 spike bins over the 1.0 s context window
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        "<details open><summary style='cursor:pointer'>"
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
# time axis, so panning or zooming any panel moves all of them in lockstep: the
# spiking of all neurons (top, with units subsampled), the movement intervals,
# and three behavioral signals. This is the raw material each training sample
# is later carved from.
#
# The full session runs ~65 min; this overview intentionally covers only the
# first 10 min, so the raster and the 50 Hz traces stay dense enough to be
# useful once you zoom in. Use the toolbar (wheel zoom, box zoom, pan, reset)
# to drill into that 10 min window; zoom in later to see a training window at
# full resolution.

# %%
#| echo: false
# Small, reusable Bokeh helpers in the same style as the torch-brain tutorials:
# one for an event raster, one for a labelled-interval strip, one for a line.
# Each accepts an x_range so several panels can be locked to one shared time axis.
import bokeh.embed.bundle as _bokeh_bundle  # noqa: E402
from bokeh.embed import file_html
from bokeh.layouts import column, gridplot
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


def _always_bundle(_objs: object) -> bool:
    return True


_bokeh_bundle._use_widgets = _always_bundle
_bokeh_bundle._use_tables = _always_bundle

try:
    import google.colab  # noqa: F401

    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    from bokeh.io import output_notebook
    from bokeh.io import show as _bokeh_show

    output_notebook(hide_banner=True)


def show(layout, title="torch-brain tutorial figure"):
    """Render a Bokeh layout.

    On Colab, plain `bokeh.io.show` (Jupyter "notebook comms": BokehJS loads
    once, in whichever cell runs first, and every later `show()` call just
    assumes that copy is still on the page) works fine, since Colab is a real
    Jupyter frontend. Using it there also avoids a DOM race between Colab's
    sandboxed output iframe and a giant inline BokehJS+figure HTML blob,
    which otherwise sometimes surfaces as a harmless but noisy `replaceChild`
    error in the browser console.

    Elsewhere, that notebook-comms assumption breaks in frontends that
    sandbox each cell's output separately (e.g. VS Code's Jupyter
    extension), where the div silently stays empty. So `file_html` bundles
    BokehJS with the figure into one self-contained blob (~4 MB, via
    `resources=INLINE`) instead, so each cell renders on its own regardless
    of frontend, and the same HTML also drops into the rendered GitHub Pages
    site with no CDN needed at view time.

    Note the module-level patch above: it forces each blob to carry the full
    library so widget cells still render when several blobs share one page
    (only relevant to the non-Colab, `file_html` path).
    """
    if _IN_COLAB:
        _bokeh_show(layout)
    else:
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
    """Round a raw tick step up to a clean 1/2/2.5/5 x 10^n value.

    Includes 2.5 (not just 1/2/5/10) so the rounded ceiling doesn't overshoot
    the actual max by up to 2x when raw_step lands just above 2 x 10^n.
    """
    if raw_step <= 0:
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    base = 10.0**exponent
    for m in (1, 2, 2.5, 5, 10):
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
#| echo: false
import gc
from types import SimpleNamespace

# A fresh handle just for this overview (all reads below are lazy).
ov_rec = BaseDataset(
    dataset_dir=os.path.join(DATA_ROOT, DATASET_DIRNAME),
    recording_ids=[RECORDING_ID],
).get_recording(RECORDING_ID)

T_END = float(ov_rec.domain.end[-1])
OV_WINDOW_S = 600.0  # overview covers only the first 10 min of the ~65 min session
ov_end = min(OV_WINDOW_S, T_END)
spk_t = np.asarray(ov_rec.spikes.timestamps)
spk_u = np.asarray(ov_rec.spikes.unit_index)
n_units = len(ov_rec.units.id)

# Restrict to the first OV_WINDOW_S seconds before thinning, so the glyph
# budget below buys resolution over 10 min instead of the whole session.
win = spk_t < ov_end
spk_t, spk_u = spk_t[win], spk_u[win]

# Bokeh draws every point in the browser (no rasterization). Over just 10 min
# the point counts are small enough that we set the budget above the actual
# count, so stride below resolves to 1: no thinning, full resolution.
GLYPH_BUDGET = 500_000
keep = np.arange(0, n_units, max(1, n_units // 70))
m = np.isin(spk_u, keep)
sub_t = spk_t[m]
sub_row = np.searchsorted(keep, spk_u[m])  # kept unit id -> compact row index
stride = max(1, len(sub_t) // GLYPH_BUDGET)
raster = SimpleNamespace(timestamps=sub_t[::stride], unit_index=sub_row[::stride])


def _thin(obj, field, target=100_000, window_s=ov_end):
    """Stride a 50 Hz signal down to ~target points, restricted to the first `window_s` s.

    Over a 10 min window a 50 Hz signal is only ~30k samples, well under
    `target`, so stride resolves to 1: this is full resolution, not thinning.
    """
    ts = np.asarray(obj.timestamps)
    y = np.asarray(getattr(obj, field))
    w = ts < window_s
    ts, y = ts[w], y[w]
    s = max(1, len(ts) // target)
    sh = SimpleNamespace(timestamps=ts[::s], domain=obj.domain)
    setattr(sh, field, y[::s])
    return sh


# One shared time range links every panel: Bokeh's equivalent of sharex=True.
# Times are passed in milliseconds because the helpers use a datetime x axis.
# Start zoomed in on the minute 5-6 slice of the 10 min overview window;
# bounds keeps pan/zoom from leaving that window, and the reset tool snaps
# back to this default view.
DEFAULT_ZOOM_START_S = 300  # 5 min
DEFAULT_ZOOM_END_S = 360  # 6 min
shared_x = Range1d(
    min(DEFAULT_ZOOM_START_S, ov_end) * 1e3,
    min(DEFAULT_ZOOM_END_S, ov_end) * 1e3,
    bounds=(0.0, ov_end * 1e3),
)
W = 700

p_raster = plot_spikes(raster, x_range=shared_x, width=W, height=220)
p_raster.title.text = (
    f"First {ov_end / 60:.0f} min of the session ({len(keep)} of {n_units} neurons)"
)

_mi = ov_rec.task_aligned_intervals.movement_intervals
_mi_mask = np.asarray(_mi.start) < ov_end
movement_ov = SimpleNamespace(
    start=np.asarray(_mi.start)[_mi_mask], end=np.asarray(_mi.end)[_mi_mask]
)
p_movement = plot_intervals(
    movement_ov,
    x_range=shared_x,
    title="movement intervals",
    width=W,
    height=45,
)

p_wheel = plot_time_series(
    _thin(ov_rec.wheel, "speed"),
    "speed",
    x_range=shared_x,
    y_axis_label="wheel speed",
    width=W,
    height=80,
)
p_whisk = plot_time_series(
    _thin(ov_rec.whisker, "motion_energy"),
    "motion_energy",
    x_range=shared_x,
    y_axis_label="whisker ME",
    width=W,
    height=80,
)
p_paw = plot_time_series(
    _thin(ov_rec.paws, "left_paw_speed"),
    "left_paw_speed",
    x_range=shared_x,
    y_axis_label="L paw speed",
    width=W,
    height=115,
)

# Only the bottom panel needs to show the (shared) time axis.
for p in (p_raster, p_movement, p_wheel, p_whisk):
    p.xaxis.visible = False
p_paw.xaxis.axis_label = "time in session"

# Drop the y axis line, tick marks, and numbers on the signal panels; the
# axis label (e.g. "wheel speed") already says what each panel is. An empty
# ticker (rather than hiding labels via font size) guarantees no tick or
# number is drawn at all.
for p in (p_wheel, p_whisk, p_paw):
    p.yaxis.axis_line_color = None
    p.yaxis.ticker = FixedTicker(ticks=[])
    p.ygrid.grid_line_color = None

# One merged toolbar for the whole column instead of one per panel: five
# repeated toolbars ate up extra width and pushed the figure past the page's
# content column.
show(
    gridplot(
        [[p_raster], [p_movement], [p_wheel], [p_whisk], [p_paw]],
        toolbar_location="right",
        merge_tools=True,
    )
)

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
#| code-fold: show
from pathlib import Path
from typing import Literal

from torch_brain.datasets import Dataset, DatasetIndex, SpikingDatasetMixin
from torch_brain.utils import bin_spikes


class IBLBrainWideMap2025(SpikingDatasetMixin, Dataset):
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
        recording = self.get_recording(self.recording_id)

        intervals = recording.task_aligned_intervals.movement_intervals
        intervals = intervals & getattr(recording, f"{self.split}_domain")
        intervals = intervals & recording.wheel._domain
        return {self.recording_id: intervals}

    # `index` is a DatasetIndex(recording_id, start, end) produced by the sampler.
    def __getitem__(self, index: DatasetIndex):
        recording = self.get_recording(index.recording_id)
        data = recording.slice(index.start, index.end)

        # All models take (num_bins, num_units) and return (out_samples, out_dim).

        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        X = torch.from_numpy(X).float()

        Y = np.asarray(data.wheel.speed, dtype=np.float32)  # shape: (out_samples,)
        Y = torch.from_numpy(Y).unsqueeze(-1)

        return X, Y


# %% [markdown]
# ## Choosing a Sampler
#
# The `Dataset` decides *where* sampling is allowed (`get_sampling_intervals`)
# and how a window becomes `(X, y)` (`__getitem__`). It says nothing about *what
# counts as one window* or *in what order* windows are drawn: that is the
# sampler's job, and TorchBrain ships several in `torch_brain.samplers`. Below we
# build the two most common, hand each the very same `Dataset`, and contrast
# them:
#
# - **`TrialSampler`** treats every interval it is given as one complete sample.
#   Fed the tight, 1.0 s movement windows from `get_sampling_intervals`, it emits
#   exactly one window per trial, each locked to a movement onset. This is the
#   trial-based paradigm: a sample *is* a task event.
# - **`RandomFixedWindowSampler`** ignores trial structure. Given a broad,
#   continuous interval (here the whole training block) and a `window_length`, it
#   carves as many fixed-length windows as fit, each at a fresh **random offset**,
#   and re-draws those offsets every epoch. This is the self-supervised /
#   pretraining paradigm: sample the recording *everywhere*, not just at task
#   events.
#
# The real difference is *who sets the window boundaries*: with `TrialSampler`
# the **Dataset** does, upstream in `get_sampling_intervals`; with
# `RandomFixedWindowSampler` the **sampler** does, via `window_length`. That is
# also why we hand them different `sampling_intervals`: the tight per-trial
# windows for one, the broad continuous `train_domain` for the other.

# %%
from torch_brain.data import Interval
from torch_brain.samplers import RandomFixedWindowSampler, TrialSampler

# One lazy Dataset handle for this section (the training pipeline below builds
# its own train/val/test handles).
demo_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="train", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
demo_rec = demo_ds.get_recording(RECORDING_ID)

# (1) Trial-aligned: one window per movement trial. get_sampling_intervals()
#     already cut the tight 1.0 s intervals, so the sampler just hands each back.
trial_intervals = demo_ds.get_sampling_intervals()
trial_sampler = TrialSampler(sampling_intervals=trial_intervals, shuffle=True)

# (2) Random fixed windows: carve CONTEXT_WINDOW-long windows at random offsets
#     from the *continuous* training block, re-jittered every epoch. We seed the
#     RNG only so this notebook renders the same picture every time.
broad_intervals = {RECORDING_ID: demo_rec.train_domain & demo_rec.wheel._domain}
random_sampler = RandomFixedWindowSampler(
    sampling_intervals=broad_intervals,
    window_length=IBLBrainWideMap2025.CONTEXT_WINDOW,  # 1.0 s
    generator=torch.Generator().manual_seed(3),
)

session_s = float(demo_rec.domain.end[-1])
block_s = float(
    np.sum(
        np.asarray(broad_intervals[RECORDING_ID].end)
        - np.asarray(broad_intervals[RECORDING_ID].start)
    )
)
trial_cov_s = len(trial_sampler) * 1.0  # each movement window is 1.0 s
rand_cov_s = len(random_sampler) * random_sampler.window_length
print(
    f"TrialSampler:             {len(trial_sampler):4d} windows / epoch  "
    f"(one per trial, at movement onsets)"
)
print(
    f"RandomFixedWindowSampler: {len(random_sampler):4d} windows / epoch  "
    f"(tiling the {block_s:.0f}s training block)"
)
print()
print(f"session length: {session_s:.0f}s")
print(
    f"  trial windows  cover {trial_cov_s:6.0f}s = "
    f"{100 * trial_cov_s / session_s:4.1f}% of the session"
)
print(
    f"  random windows cover {rand_cov_s:6.0f}s = "
    f"{100 * rand_cov_s / session_s:4.1f}% of the session"
)

# %% [markdown]
# The figure below makes the contrast concrete. The top pair spans the whole
# ~809 s training block: the continuous wheel-speed signal, then the windows each
# sampler emits (`TrialSampler` in red, then `RandomFixedWindowSampler` for two
# separate epochs, in blue and green). At this scale the story is *coverage*: a
# sparse scatter of trial windows against a near-solid tiling. The bottom pair
# zooms to ~40 s, where two more things become visible: trial windows sit right
# where the wheel speed lifts off (the movement onsets), while random windows
# tile straight across regardless; and the blue and green rows are offset from
# each other, the per-epoch jitter that doubles as free data augmentation.

# %%
#| code-fold: true
#| code-summary: "Bokeh: TrialSampler vs RandomFixedWindowSampler windows (full block + zoom)"
from types import SimpleNamespace

from bokeh.models import Div


def _windows_as_interval(sampler):
    """The windows a sampler emits this epoch, as a time-sorted Interval.

    The sampler yields (shuffled) DatasetIndex objects; for drawing we only need
    their start/end, sorted so the strip reads left to right.
    """
    idx = list(sampler)
    starts = np.array([i.start for i in idx], dtype=float)
    ends = np.array([i.end for i in idx], dtype=float)
    order = np.argsort(starts)
    return Interval(start=starts[order], end=ends[order])


trial_win = _windows_as_interval(trial_sampler)
rand_win_1 = _windows_as_interval(random_sampler)  # epoch 1 offsets
rand_win_2 = _windows_as_interval(random_sampler)  # epoch 2 offsets (re-jittered)

tb_start = float(np.asarray(demo_rec.train_domain.start)[0])
tb_end = float(np.asarray(demo_rec.train_domain.end)[-1])

# Wheel-speed context, read once at full resolution. We mask by absolute time
# rather than slice(), because slice() re-bases timestamps to 0.
wheel_ts = np.asarray(demo_rec.wheel.timestamps)
wheel_sp = np.asarray(demo_rec.wheel.speed)


def _wheel_view(t0, t1, target=3000):
    """Wheel speed over [t0, t1] (absolute seconds), thinned to ~target points."""
    m = (wheel_ts >= t0) & (wheel_ts <= t1)
    ts, sp = wheel_ts[m], wheel_sp[m]
    step = max(1, len(ts) // target)
    return SimpleNamespace(
        timestamps=ts[::step],
        speed=sp[::step],
        domain=Interval(start=np.array([t0]), end=np.array([t1])),
    )


W = 700


def _windows_fig(x_range, height):
    """Three stacked strips: TrialSampler, then two RandomFixedWindow epochs."""
    return plot_intervals(
        trial_win,
        rand_win_1,
        rand_win_2,
        x_range=x_range,
        width=W,
        height=height,
    )


legend = Div(
    text=(
        "<span style='color:red'>&#9632;</span> TrialSampler &nbsp;&nbsp;"
        "<span style='color:blue'>&#9632;</span> RandomFixedWindow (epoch 1) &nbsp;&nbsp;"
        "<span style='color:green'>&#9632;</span> RandomFixedWindow (epoch 2)"
    )
)

# --- full training block: sparse trials vs. dense tiling ---
full_x = Range1d(tb_start * 1e3, tb_end * 1e3, bounds=(tb_start * 1e3, tb_end * 1e3))
p_wheel_full = plot_time_series(
    _wheel_view(tb_start, tb_end),
    "speed",
    x_range=full_x,
    y_axis_label="wheel speed",
    width=W,
    height=110,
)
p_wheel_full.title.text = (
    f"Whole training block ({tb_end - tb_start:.0f}s): sparse trials vs. dense tiling"
)
p_wheel_full.xaxis.visible = False
p_win_full = _windows_fig(full_x, height=110)
p_win_full.xaxis.axis_label = "time in session"

# --- ~40 s zoom, centered on a trial-dense stretch ---
tw_starts = np.sort(np.asarray(trial_win.start))
anchor = float(tw_starts[len(tw_starts) // 3])
z0 = max(tb_start, anchor - 3.0)
z1 = min(tb_end, z0 + 40.0)
zoom_x = Range1d(z0 * 1e3, z1 * 1e3, bounds=(z0 * 1e3, z1 * 1e3))
p_wheel_zoom = plot_time_series(
    _wheel_view(z0, z1),
    "speed",
    x_range=zoom_x,
    y_axis_label="wheel speed",
    width=W,
    height=120,
)
p_wheel_zoom.title.text = (
    f"Zoom ({z1 - z0:.0f}s): trial windows align to movement onsets; "
    f"random windows do not"
)
p_wheel_zoom.xaxis.visible = False
p_win_zoom = _windows_fig(zoom_x, height=120)
p_win_zoom.xaxis.axis_label = "time in session"

show(column(legend, p_wheel_full, p_win_full, p_wheel_zoom, p_win_zoom))

# %% [markdown]
# For this tutorial we decode a **task-defined event**, the 1.0 s of wheel motion
# right after each movement onset, so `TrialSampler` is the natural choice: every
# sample is one clean, behavior-aligned trial. The training pipeline below
# therefore wraps each split in a `TrialSampler`. `RandomFixedWindowSampler`
# earns its place when you want to model the whole recording rather than isolated
# events, for example self-supervised pretraining of a foundation model, which is
# the direction the appendix points toward.

# %% [markdown]
# ## Building the train, validation, and test pipeline
#
# `IBLBrainWideMap2025` is instantiated once per split, each wrapped in its
# own `TrialSampler` and `DataLoader`. Once these are built, we will look at
# what the split boundaries and a single sample actually look like, and step
# through a real training epoch's sample order interactively.

# %%
#| code-fold: show
from torch.utils.data import DataLoader  # standard PyTorch loader
from torch_brain.samplers import TrialSampler

train_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="train", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
# We want to sample "one-trial-at-a-time", so we use the TrialSampler
train_sampler = TrialSampler(
    sampling_intervals=train_ds.get_sampling_intervals(), shuffle=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)

# Validation Dataset, Sampler, and DataLoader
val_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="val", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
val_sampler = TrialSampler(
    sampling_intervals=val_ds.get_sampling_intervals(), shuffle=False
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler)


# Test Dataset, Sampler, and DataLoader.
test_ds = IBLBrainWideMap2025(
    DATA_ROOT, split="test", bin_size=BIN_SIZE, recording_id=RECORDING_ID
)
test_sampler = TrialSampler(
    sampling_intervals=test_ds.get_sampling_intervals(), shuffle=False
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler)

# %%
print(f"Number of units: {train_ds.num_units}")
print(f"Number of training samples: {len(train_sampler)}")
print(f"Number of validation samples: {len(val_sampler)}")
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
# The three sliders below freeze one such epoch order per split (train, val,
# test): step through any of them to see the corresponding window highlighted,
# in that split's color, on the row above, and the `(X, Y)` pair, binned spikes
# and wheel speed, it turns into.

# %%
#| echo: false
from bokeh.layouts import row as bokeh_row
from bokeh.models import BoxAnnotation, CustomJS, Div, Slider, Span
from bokeh.palettes import Greys256

recording = train_ds.get_recording(RECORDING_ID)
split_t_end = float(recording.domain.end[-1])
split_x_range = Range1d(0.0, split_t_end * 1e3, bounds=(0.0, split_t_end * 1e3))
W = 700

domains_fig = plot_intervals(
    recording.train_domain,
    recording.val_domain,
    recording.test_domain,
    x_range=split_x_range,
    title="train / val / test domain: one contiguous block each",
    width=W,
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
    width=W,
    height=90,
)
samples_fig.xaxis.axis_label = "time in session"

legend = Div(
    text=(
        "<div style='font-size:18px'>"
        "<span style='color:red'>&#9632;</span> train &nbsp;&nbsp;"
        "<span style='color:blue'>&#9632;</span> val &nbsp;&nbsp;"
        "<span style='color:green'>&#9632;</span> test"
        "</div>"
    )
)

# One interactive column per split (train / val / test), side by side. Each
# column has its own slider stepping through that split's samples, plus the
# (X, Y) pair, and drops a color-matched highlight box + marker on the split's
# own row of the samples_fig strip above (train=red row 0, val=blue row 1,
# test=green row 2, matching plot_intervals' row order and colors).
# Each tuple: (name, dataset, sampler, strip color, strip row, light slider color).
SPLITS = [
    ("train", train_ds, train_sampler, "red", 0, "lightcoral"),
    ("val", val_ds, val_sampler, "blue", 1, "lightblue"),
    ("test", test_ds, test_sampler, "green", 2, "lightgreen"),
]

# CONTEXT_WINDOW / out_samples are identical across splits, so the target-time
# axis for the wheel (Y) line is shared.
t_local = np.linspace(0.0, train_ds.CONTEXT_WINDOW, train_ds.out_samples).tolist()

col_w = W // 3  # three columns share the width of the strips above
split_columns = []
for name, ds, sampler, color, row_i, light_color in SPLITS:
    # Precompute this split's whole epoch of (X, Y) pairs so each slider only
    # ever swaps already-computed arrays, no HDF5 reads. Keep each raster as a
    # 2D numpy array, NOT `.tolist()`: Bokeh's `image` glyph needs a real 2D
    # array (a nested Python list serializes as undefined and kills the render).
    indices = list(sampler)
    n = len(indices)
    Xs, Ys = [], []
    for idx in indices:
        Xd, Yd = ds[idx]
        Xs.append(Xd.T.numpy())  # (num_units, num_bins) 2D array for the image
        Ys.append(Yd[:, 0].numpy().tolist())  # (out_samples,) 1D list for a line
    starts = [float(idx.start) for idx in indices]
    ends = [float(idx.end) for idx in indices]

    # The moving highlight + marker for this split's row on the strip above.
    highlight = BoxAnnotation(
        left=starts[0] * 1e3,
        right=ends[0] * 1e3,
        bottom=-row_i - 0.45,
        top=-row_i + 0.45,
        fill_color=color,
        fill_alpha=0.35,
    )
    samples_fig.add_layout(highlight)
    sample_marker = Span(
        location=(starts[0] + ends[0]) * 0.5 * 1e3,
        dimension="height",
        line_color=color,
        line_width=2,
    )
    samples_fig.add_layout(sample_marker)

    # X: binned-spike raster.
    raster_source = ColumnDataSource(data=dict(image=[Xs[0]]))
    p_raster = figure(
        width=col_w,
        height=220,
        title=f"{name}: binned spikes (X)",
        x_axis_label="Time bin",
        y_axis_label="Unit",
        toolbar_location=None,
    )
    p_raster.image(
        image="image",
        x=0,
        y=0,
        dw=ds.num_bins,
        dh=ds.num_units,
        source=raster_source,
        # Greys256 runs black -> white; reversed so higher firing rate reads darker.
        palette=list(reversed(Greys256)),
    )
    p_raster.yaxis.visible = False
    p_raster.grid.grid_line_color = None

    # Y: wheel speed.
    wheel_source = ColumnDataSource(data=dict(x=t_local, y=Ys[0]))
    p_wheel = figure(
        width=col_w,
        height=180,
        title=f"{name}: wheel speed (Y)",
        x_axis_label="Time within window (s)",
        y_axis_label="Wheel speed",
        toolbar_location=None,
    )
    p_wheel.line("x", "y", source=wheel_source, line_width=2, color="black")
    p_wheel.yaxis.visible = False
    p_wheel.grid.grid_line_color = None

    slider = Slider(
        start=0,
        end=n - 1,
        value=0,
        step=1,
        title=f"{name} sample index",
        width=col_w - 20,
        bar_color=light_color,
    )
    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                slider=slider,
                raster_source=raster_source,
                wheel_source=wheel_source,
                highlight=highlight,
                sample_marker=sample_marker,
                X_list=Xs,
                Y_list=Ys,
                starts=starts,
                ends=ends,
                t_local=t_local,
            ),
            code="""
    const i = slider.value
    raster_source.data = {image: [X_list[i]]}
    wheel_source.data = {x: t_local, y: Y_list[i]}
    highlight.left = starts[i] * 1e3
    highlight.right = ends[i] * 1e3
    sample_marker.location = (starts[i] + ends[i]) * 0.5 * 1e3
    """,
        ),
    )

    split_columns.append(column(slider, p_raster, p_wheel))

show(
    column(
        legend,
        domains_fig,
        samples_fig,
        bokeh_row(*split_columns),
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
#| code-fold: show
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
#| code-fold: show
from sklearn.metrics import r2_score

optim = torch.optim.AdamW(model.parameters(), lr=LR)

val_r2_history = []

for _epoch in (epoch_pbar := tqdm(range(EPOCHS))):
    model.train()
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        model.eval()
        preds, targets = [], []
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
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
        X, Y = X, Y = X.to(device), Y.to(device)
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
