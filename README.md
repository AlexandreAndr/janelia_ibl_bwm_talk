# IBL wheel-speed minimal example (self-contained)

A transparent, standalone training example: decode **wheel speed** (1D, 50 Hz)
from whole-brain spiking activity of a single IBL session, with three small
decoders (Linear, GRU, TCN). Everything it needs lives in this folder.

## Contents

| Path | What it is |
|------|------------|
| `wheel_speed_minimal_example.py` | The example, as a jupytext percent-format script (source of truth for diffs). |
| `wheel_speed_minimal_example.ipynb` | The same notebook in `.ipynb` form, paired to the `.py` (see [Notebook formats](#notebook-formats)). |
| `dataset.py` | Vendored `IBLBrainWideMap2025` base dataset (copied from `src/core`). |
| `data/recording_ids.txt` | Recording-id list read by `dataset.py`, holding the single session used here. |
| `ibl_brain_wide_map_2025/` | The brainset pipeline, copied as-is. Its `recording_ids.txt` holds only the single session used here. |
| `processed/` | Where the pipeline writes the processed `.h5` (download target). |
| `requirements.txt` | Pinned dependency list for this folder, installable with plain `pip`. |
| `jupytext.toml` | Pairing config that keeps the `.py` and `.ipynb` in sync. |

The script imports the vendored `dataset.py` (no dependency on the repo's
`src/core`) and reads the session from the local `processed/` directory.

## Environment

This folder has its own isolated virtualenv, independent of the repo's top-level
`venv`. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Running the notebook in Google Colab instead? The first cell clones this repo
and installs `requirements.txt` for you: just click the "Open In Colab" badge
at the top of the notebook.

## Download the session

The pipeline processes exactly the single eid listed in
`ibl_brain_wide_map_2025/recording_ids.txt`
(`0802ced5-33a3-405e-8336-b65ebc5cb07c`). From this folder:

```bash
brainsets prepare ./ibl_brain_wide_map_2025 --local \
  --processed-dir ./processed --raw-dir ./raw
```

This downloads the raw IBL session via the ONE API and processes it into
`processed/ibl_brain_wide_map_2025/0802ced5-33a3-405e-8336-b65ebc5cb07c.h5`
with no unit QC filtering by default (the script applies its own `UnitFilter`).

Requirements:
- ONE API credentials configured (`~/.one/`).
- A few GB free disk: the raw download for this session is ~5.5 GB (kept in
  `raw/` unless you clean it yourself); the processed `.h5` is ~1.8 GB.

To process a different session instead, put its eid on a line in
`ibl_brain_wide_map_2025/recording_ids.txt` and update `RECORDING_ID` in the script.

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
