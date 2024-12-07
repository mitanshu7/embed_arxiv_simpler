#!/bin/bash

# 1. Activate the conda environment
source /home/$USER/miniforge3/bin/activate papermatch

# 2. Run the Python script
python /home/$USER/embed_arxiv_simpler/prepare_milvus.py >> /home/$USER/embed_arxiv_simpler/prepare_milvus.log 2>&1

# 3. Restart the systemd service
systemctl --user restart papermatch.service