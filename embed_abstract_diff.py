# Import required libraries
from sentence_transformers import SentenceTransformer # For embedding the text
import torch # For gpu 
import pandas as pd # Data manipulation
from time import time # Track time taken
from glob import glob # Gather files
import os # Folder and file creation
from tqdm import tqdm # Progress bar
tqdm.pandas() # Progress bar for pandas

################################################################################

# Track time
start = time()

# Year to diff embed
year = '24'

repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"


# Gather split file
data_folder = 'data'
split_folder = f'{data_folder}/arxiv-metadata-oai-snapshot-trim-split'
split_file = f'{split_folder}/{year}.parquet'
previous_embed = f'{repo_id}/data/{year}.parquet'

# Create folder
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
arxiv_metadata_trimmed_split = pd.read_parquet(split_file)

# Load previous_embed
print(f"Loading previous embed file: {previous_embed}")   
arxiv_previous_embed = pd.read_parquet(previous_embed)

# Merge the two dataframes on id
diff = pd.merge(arxiv_previous_embed, arxiv_metadata_trimmed_split, how='outer', on='id')

# Keep only NaN values
diff = diff[diff.isna().any(axis=1)]

# Selecting id and abstract to retain
selected_columns = ['id', 'abstract']

# Keep columns of interest
diff = diff[selected_columns]


# Create a column for embeddings
print(f"Creating embeddings for: {len(diff)} entries")
diff["vector"] = diff["abstract"].progress_apply(embed)

# Selecting id and vector to retain
selected_columns = ['id', 'vector']

# Save the embedded file
embed_filename = f'{embed_folder}/{os.path.basename(split_file)}'
print(f"Saving embedded dataframe to: {embed_filename}")
diff[selected_columns].to_parquet(embed_filename)

# Track time
toc = time()
print(f"Processed in: {(toc-tic)/60} minutes")
    

################################################################################

# Track time
end = time()

print('#'*80)
print(f"It took a total of: {(end - start)/3600} hours")
print("Done!")