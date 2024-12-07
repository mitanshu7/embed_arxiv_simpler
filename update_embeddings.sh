#!/bin/bash

# 0. Echo date
echo "Starting script at $(date)"

# 1. Activate the conda environment
source /home/$USER/miniforge3/bin/activate papermatch

# 2. Run the Python script
python /home/$USER/embed_arxiv_simpler/update_embeddings.py >> /home/$USER/embed_arxiv_simpler/update_embeddings.log 2>&1

# 3. Echo date
echo "Finished script at $(date)"