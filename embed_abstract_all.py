# Import required libraries
from sentence_transformers import SentenceTransformer # For embedding the text
import torch # For gpu 
import pandas as pd # Data manipulation
import json # To make milvus compatible $meta
from time import time # Track time taken
from glob import glob # Gather files
import os # Folder and file creation
from tqdm import tqdm # Progress bar
tqdm.pandas() # Progress bar for pandas

################################################################################

# Track time
start = time()

# Gather split files
data_folder = 'data'
split_folder = f'{data_folder}/arxiv-metadata-oai-snapshot-split'
split_files = glob(f'{split_folder}/*.parquet')
print(f"Found {len(split_files)} split files")

# Create folder
embed_folder = f"{split_folder}-embed"
os.makedirs(embed_folder, exist_ok=True)

################################################################################

# Make the app device agnostic
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "mixedbread-ai/mxbai-embed-large-v1"

# Load a pretrained Sentence Transformer model and move it to the appropriate device
print(f"Loading model {model_name} to device: {device}")
model = SentenceTransformer(model_name)
model = model.to(device)

# Function that does the embedding
def embed(input_text):
    
    # Calculate embeddings by calling model.encode(), specifying the device
    embedding = model.encode(input_text, device=device)

    return embedding

################################################################################

# Loop through each split file
for split_file in split_files:

    print('#'*80)

    # Track time
    tic = time()

    # Load metadata
    print(f"Loading metadata file: {split_file}")   
    arxiv_metadata_split = pd.read_parquet(split_file)

    # Create a column for embeddings
    print(f"Creating embeddings for: {len(arxiv_metadata_split)} entries")
    arxiv_metadata_split["vector"] = arxiv_metadata_split["abstract"].progress_apply(embed)

    # Rename columns
    arxiv_metadata_split.rename(columns={'title': 'Title', 'authors': 'Authors', 'abstract': 'Abstract'}, inplace=True)

    # Add URL column
    arxiv_metadata_split['URL'] = 'https://arxiv.org/abs/' + arxiv_metadata_split['id']

    # Create milvus compatible parquet file, $meta is a json string of the metadata
    arxiv_metadata_split['$meta'] = arxiv_metadata_split[['Title', 'Authors', 'Abstract', 'URL']].apply(lambda row: json.dumps(row.to_dict()), axis=1)
    
    # Selecting id, vector and $meta to retain
    selected_columns = ['id', 'vector', '$meta']

    # Save the embedded file
    embed_filename = f'{embed_folder}/{os.path.basename(split_file)}'
    print(f"Saving embedded dataframe to: {embed_filename}")
    # Keeping index=False to avoid saving the index column as a separate column in the parquet file
    # This keeps milvus from throwing an error when importing the parquet file
    arxiv_metadata_split[selected_columns].to_parquet(embed_filename, index=False)

    # Track time
    toc = time()
    print(f"Processed in: {(toc-tic)/60} minutes")
    

################################################################################

# Track time
end = time()

print('#'*80)
print(f"It took a total of: {(end - start)/3600} hours")
print("Done!")