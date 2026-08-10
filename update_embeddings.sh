#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# 0. Echo date
echo "Starting script at $(date)"

# 1. Activate the python environment
source .venv/bin/activate

# 2. Run the Python script
python update_embeddings.py >> update_embeddings.log 2>&1
python update_zilliz.py >> update_zilliz.log 2>&1

# 3. Echo date
echo "Finished script at $(date)"