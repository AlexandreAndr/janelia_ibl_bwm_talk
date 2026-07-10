#!/usr/bin/env bash
# One command to go from an edited .py/.ipynb to a live Quarto preview:
#   1. jupytext --sync   (bring .py and .ipynb back in line)
#   2. nbconvert --execute --inplace   (re-run the notebook, bake in fresh outputs)
#   3. build.sh --preview   (render + serve on localhost:8080, isolated from the .py)
#
# Usage:
#   ./preview.sh              # sync, re-execute, preview
#   ./preview.sh --no-execute # sync only, skip re-running the notebook (outputs may be stale)
set -euo pipefail
cd "$(dirname "$0")"

NB=demo.ipynb
PY=demo.py
PYTHON="$PWD/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating .venv and installing requirements-dev.txt ..."
  uv venv .venv
  uv pip install -p "$PYTHON" -r requirements-dev.txt
fi

"$PWD/.venv/bin/jupytext" --sync "$PY"

if [[ "${1:-}" != "--no-execute" ]]; then
  QUARTO_PYTHON="$PYTHON" "$PWD/.venv/bin/jupyter" nbconvert --to notebook --execute --inplace "$NB"
fi

exec ./build.sh --preview
