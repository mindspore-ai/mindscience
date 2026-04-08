#!/bin/bash
set -euo pipefail

port="${1:-8888}"

# Activate the site module or Python environment before launch if needed.
# module load python
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate notebook-env

jupyter lab --no-browser --ip=127.0.0.1 --port="$port"
