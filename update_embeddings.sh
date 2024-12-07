#!/bin/bash

# 1. Activate the conda environment
source /home/$USER/miniforge3/bin/activate papermatch

# 2. Run the Python script
python /home/$USER/embed_arxiv_simpler/update_embeddings.py >> /home/$USER/embed_arxiv_simpler/update_embeddings.log 2>&1