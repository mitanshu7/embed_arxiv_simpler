# Import required libraries
from glob import glob

from datasets import load_dataset
from huggingface_hub import snapshot_download  # To download and concatenate vector data

# Whether to consolidat FLoat or Binary vectors
FLOAT = False
# Also, whether to split the consolidated file into parts
# Set 0 for no sharding.
num_parts = 0

# When 100% of the data is float, use the float repo
if FLOAT:
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

# When 100% of the data is binary, use the binary repo
else:
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

# Download the repo
repo_type = "dataset"
local_dir = repo_id
allow_patterns = f"{repo_id}/data/*.parquet"

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
dataset = load_dataset("parquet", data_dir=f"{repo_id}/data/")

# Save consolidated dataset
dataset["train"].to_parquet(f"{repo_id}/consolidated.parquet")


# Save consolidated dataset in parts
if num_parts:
    print(f"Splitting consolidated dataset in {num_parts} parts.")

    for i in range(num_parts):
        dataset["train"].shard(num_parts, index=i).to_parquet(
            f"{repo_id}/part_{i + 1}.parquet"
        )
