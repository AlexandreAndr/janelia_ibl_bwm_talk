# IBL wheel-speed minimal example (self-contained)

A transparent, standalone training example: decode **wheel speed** (1D, 50 Hz)
from whole-brain spiking activity of a single IBL session, with three small
decoders (Linear, GRU, TCN). Everything it needs lives in this folder.

## Contents

| Path | What it is |
|------|------------|
| `wheel_speed_minimal_example.py` | The example, as a jupytext percent-format script (source of truth for diffs). |
| `wheel_speed_minimal_example.ipynb` | The same notebook in `.ipynb` form, paired to the `.py` (see [Notebook formats](#notebook-formats)). |
| `recording_ids.txt` | Recording-id list read by `IBLBrainWideMap2025` (defined in the script), holding the single session used here. |
| [`ibl_brain_wide_map_2025/`](https://github.com/AlexandreAndr/janelia_ibl_bwm_talk/tree/main/ibl_brain_wide_map_2025) | This tutorial's own `brainsets` pipeline, kept local rather than in the [shared community pipelines](https://github.com/neuro-galaxy/torch_brain/tree/main/torch_brain/pipeline/brainsets-pipelines) because it's just an example for this tutorial. Its `recording_ids.txt` holds only the single session used here. |
| `processed/` | Where the pipeline writes the processed `.h5` (download target). |
| `requirements.txt` | Pinned runtime dependency list, installable with plain `pip`. |
| `requirements-dev.txt` | Adds notebook/Quarto tooling (jupytext, nbconvert, ipykernel) on top of `requirements.txt`. |
| `jupytext.toml` | Pairing config that keeps the `.py` and `.ipynb` in sync. |
| `preview.sh` | One-command sync + re-execute + live Quarto preview (see [Building docs with Quarto](#building-docs-with-quarto)). |

The script defines `IBLBrainWideMap2025` itself (no dependency on the repo's
`src/core`) and reads the session from the local `processed/` directory.

## Environment

This folder has its own isolated virtualenv, independent of the repo's top-level
`venv`. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

To also edit/sync the notebook or render docs with Quarto, install the dev extras
instead (superset of `requirements.txt`):

```bash
pip install -r requirements-dev.txt
```

Running the notebook in Google Colab instead? The first cell installs the
pinned dependencies for you: just click the "Open In Colab" badge at the top
of the notebook. A later cell then downloads the pre-processed session
automatically (see [Download the session](#download-the-session)): no repo
checkout, manual steps, or ONE API credentials needed to just run the example.

## Download the session

By default, the notebook downloads the already-processed `.h5` (~0.4 GB,
already filtered to good-quality units) for
`0802ced5-33a3-405e-8336-b65ebc5cb07c` straight from the
[`AlexAndreUpenn/ibl-bwm-wheel-speed-demo`](https://huggingface.co/datasets/AlexAndreUpenn/ibl-bwm-wheel-speed-demo)
dataset on the Hugging Face Hub, into
`processed/ibl_brain_wide_map_2025/0802ced5-33a3-405e-8336-b65ebc5cb07c.h5`.
This is what runs on Colab and is the fastest way to get started: it skips
the raw IBL download and the `brainsets` processing step entirely, and needs
no credentials since the file is public. The notebook skips this download if
the file already exists locally (e.g. from running the pipeline yourself).

To (re)build the `.h5` from raw data yourself instead (e.g. to process a
different session, or to pick up pipeline changes), the pipeline processes
exactly the eid(s) listed in `ibl_brain_wide_map_2025/recording_ids.txt`. From
this folder:

```bash
brainsets prepare ./ibl_brain_wide_map_2025 --local \
  --processed-dir ./processed --raw-dir ./raw \
  --unit-filter probe_qc --unit-filter firing_rate --unit-filter unit_qc
```

This downloads the raw IBL session via the ONE API and processes it into
`processed/ibl_brain_wide_map_2025/0802ced5-33a3-405e-8336-b65ebc5cb07c.h5`.
The three `--unit-filter` flags keep only good-quality units: `probe_qc`
(probe passed QC), `firing_rate` (> 1 Hz), and `unit_qc` (KiloSort/IBL's own
"good" label). Omit any of them to keep more units (e.g. no flags at all
processes every recorded unit, 1547 for this session, instead of the 358
good-quality ones).

Requirements:
- ONE API credentials configured (`~/.one/`).
- A few GB free disk: the raw download for this session is ~5.5 GB (kept in
  `raw/` unless you clean it yourself); the processed `.h5` is ~0.4 GB with
  the unit filters above (~1.9 GB unfiltered).

To process a different session instead, put its eid on a line in
`ibl_brain_wide_map_2025/recording_ids.txt`, update `RECORDING_ID` in the
script, and re-upload the resulting `.h5` to the Hugging Face dataset above
(or point the notebook's `hf_hub_download` call at wherever you host it) if
you also want the Colab path to pick it up.

## Run the example

```bash
python wheel_speed_minimal_example.py
```

Or open `wheel_speed_minimal_example.ipynb` in Jupyter using the `./.venv` kernel.

## Notebook formats

The `.py` (jupytext percent format) and `.ipynb` are **paired** via
`jupytext.toml`. The `.py` is the source of truth for review and diffs; the
`.ipynb` is the runnable/renderable copy. After editing either, sync the other:

```bash
jupytext --sync wheel_speed_minimal_example.py    # or the .ipynb
```

Regenerate the `.ipynb` from scratch if needed:

```bash
jupytext --to ipynb wheel_speed_minimal_example.py
```

## Building docs with Quarto

[Quarto](https://quarto.org/) (v1.9+) renders the notebook to HTML. Config lives
in `_quarto.yml`, and `execute.enabled` is `false`: Quarto renders straight from
the cell outputs already baked into the `.ipynb` (it never re-trains the model).

The notebook trains a model, so first execute it **once** to bake in its outputs,
then use `build.sh` to render:

```bash
# 1. Execute the notebook in place (populates cell outputs; ~a couple minutes).
#    Re-run this whenever the code changes, then commit the .ipynb.
QUARTO_PYTHON="$PWD/.venv/bin/python" \
  jupyter nbconvert --to notebook --execute --inplace \
  wheel_speed_minimal_example.ipynb

# 2. Build the self-contained site into ./_site (open _site/index.html).
./build.sh
```

**Why `build.sh` instead of `quarto render` directly:** Quarto renders
`foo.ipynb` from a sibling `foo.py` whenever one exists (this folder pairs them
via jupytext). That script path has no cell outputs and drops `code-fold`, so
plots and the collapsible code blocks vanish. `build.sh` sidesteps this by
rendering a copy of the `.ipynb` + `_quarto.yml` in a clean directory that has no
`.py` beside it. The same script runs in CI (see `.github/workflows/deploy.yml`).

### Launch a local preview server

`build.sh --preview` serves the doc on `localhost:8080` with live reload (again
from an isolated dir so folding is preserved):

```bash
./build.sh --preview
```

Then open <http://localhost:8080/>. To author a `.qmd` version instead,
`quarto convert wheel_speed_minimal_example.ipynb`.

### All-in-one: `preview.sh`

`preview.sh` chains the three steps above into one command: `jupytext --sync`,
re-execute the notebook, then `build.sh --preview`. It bootstraps `.venv` from
`requirements-dev.txt` via `uv` if the venv doesn't exist yet.

```bash
./preview.sh              # sync, re-execute, preview
./preview.sh --no-execute # sync only, skip re-running the notebook
```
