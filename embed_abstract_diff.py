# Import required libraries
from sentence_transformers import SentenceTransformer # For embedding the text
import torch # For gpu 
import pandas as pd # Data manipulation
from time import time # Track time taken
from huggingface_hub import snapshot_download # Download previous embeddings
import json # To make milvus compatible $meta
import os # Folder and file creation
from tqdm import tqdm # Progress bar
tqdm.pandas() # Progress bar for pandas

################################################################################

# Track time
start = time()

# Year to diff embed
year = '24'

# Setup transaction details
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"
repo_type = "dataset"
local_dir = repo_id
allow_patterns = f"data/{year}.parquet"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Download the repo
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)

# Gather previous embed file
previous_embed = f'{repo_id}/data/{year}.parquet'

# Gather split file
data_folder = 'data'
split_folder = f'{data_folder}/arxiv-metadata-oai-snapshot-split'
split_file = f'{split_folder}/{year}.parquet'

# Create embed folder
embed_folder = f"{split_folder}-diff-embed"
os.makedirs(embed_folder, exist_ok=True)

################################################################################

# Make the app device agnostic
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained Sentence Transformer model and move it to the appropriate device
print(f"Loading model to device: {device}")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
model = model.to(device)

# Function that does the embedding
def embed(input_text):
    
    # Calculate embeddings by calling model.encode(), specifying the device
    embedding = model.encode(input_text, device=device)

    return embedding

################################################################################

print('#'*80)

# Track time
tic = time()

# Load metadata
print(f"Loading metadata file: {split_file}")   
arxiv_metadata_split = pd.read_parquet(split_file)

# Load previous_embed
print(f"Loading previously embedded file: {previous_embed}")   
previous_embeddings = pd.read_parquet(previous_embed)

# Find papers that are not in the previous embeddings
new_papers = arxiv_metadata_split[~arxiv_metadata_split['id'].isin(previous_embeddings['id'])]

# Create a column for embeddings
print(f"Creating new embeddings for: {len(new_papers)} entries")
new_papers["vector"] = new_papers["abstract"].progress_apply(embed)

# Rename columns
new_papers.rename(columns={'title': 'Title', 'authors': 'Authors', 'abstract': 'Abstract'}, inplace=True)

# Add URL column
new_papers['URL'] = 'https://arxiv.org/abs/' + new_papers['id']

# Create milvus compatible parquet file, $meta is a json string of the metadata
new_papers['$meta'] = new_papers[['Title', 'Authors', 'Abstract', 'URL']].apply(lambda row: json.dumps(row.to_dict()), axis=1)

# Selecting id, vector and $meta to retain
selected_columns = ['id', 'vector', '$meta']

# Merge previous embeddings and new embeddings
new_embeddings = pd.concat([previous_embeddings, new_papers[selected_columns]])

# Save the embedded file
embed_filename = f'{embed_folder}/{year}.parquet'
print(f"Saving newly embedded dataframe to: {embed_filename}")
# Keeping index=False to avoid saving the index column as a separate column in the parquet file
# This keeps milvus from throwing an error when importing the parquet file
new_embeddings.to_parquet(embed_filename, index=False)

# Track time
toc = time()
print(f"Processed in: {(toc-tic)/60} minutes")
    

################################################################################

# Track time
end = time()

print('#'*80)
print(f"It took a total of: {(end - start)/3600} hours")
print("Done!")