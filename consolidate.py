# Import required libraries
from glob import glob

from datasets import load_dataset
from huggingface_hub import snapshot_download  # To download and concatenate vector data

FLOAT = False

# When 100% of the data is float, use the float repo
if FLOAT:
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

# When 100% of the data is binary, use the binary repo
else:
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

# Download the repo
repo_type = "dataset"
local_dir = repo_id
allow_patterns = "*.parquet"

# Download the repo
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)

# Gather the file names:
parquet_files = glob(f"{repo_id}/**/*.parquet", recursive=True)

# Create a new dataset
dataset = load_dataset("parquet", data_dir=repo_id)

# Save consolidated dataset
dataset["train"].to_parquet(f"{repo_id}/consolidated.parquet")
