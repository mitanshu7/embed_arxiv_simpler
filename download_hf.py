# Import required libraries
from huggingface_hub import snapshot_download
import os # Folder creation

# Setup transaction details
repo_id = ""
repo_type = "dataset"
local_dir = repo_id.replace('/', '_')
allow_patterns = "*.parquet"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Download the repo
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)