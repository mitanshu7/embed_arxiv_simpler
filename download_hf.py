# Import required libraries
from huggingface_hub import snapshot_download
import os # Folder creation

# Setup transaction details
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"
repo_type = "dataset"
local_dir = repo_id
allow_patterns = "*/24.parquet"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Download the repo
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)