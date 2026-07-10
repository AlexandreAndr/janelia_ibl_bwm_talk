#!/usr/bin/env bash
# Render the notebook into a self-contained site in ./_site (code folding intact).
#
# Why the isolated build dir: Quarto renders `foo.ipynb` from a sibling
# `foo.py` whenever one exists (jupytext pairing). That script path carries no
# cell outputs and does NOT apply `code-fold`, so plots and the collapsible code
# blocks disappear. We therefore render a copy of the .ipynb + _quarto.yml in a
# clean directory with no .py beside it.
#
# The notebook must already carry its executed outputs (see README: run
# `jupyter nbconvert --to notebook --execute --inplace demo.ipynb`
# once). Quarto does not execute anything here (execute.enabled: false).
#
# Usage:
#   ./build.sh            # build _site/
#   ./build.sh --preview  # build, then serve with live-ish reload on :8080
set -euo pipefail
cd "$(dirname "$0")"

NB=demo.ipynb
BUILD=.build

rm -rf "$BUILD" _site
mkdir -p "$BUILD"
cp "$NB" _quarto.yml "$BUILD/"

if [[ "${1:-}" == "--preview" ]]; then
  # Serve from the isolated dir. Output-dir is _site (relative to $BUILD).
  cd "$BUILD"
  exec quarto preview "$NB" --port 8080 --host localhost --no-browser
fi

( cd "$BUILD" && quarto render "$NB" -M embed-resources:true )
mv "$BUILD/_site" _site
# project.type: default renders to <name>.html; serve it at the site root too.
cp _site/demo.html _site/index.html
rm -rf "$BUILD"
echo "Built _site/ (open _site/index.html)"
