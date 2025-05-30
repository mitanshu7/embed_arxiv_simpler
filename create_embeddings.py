## Download the arXiv metadata from Kaggle
## https://www.kaggle.com/datasets/Cornell-University/arxiv

## Requires the Kaggle API to be installed
## Using subprocess to run the Kaggle CLI commands instead of Kaggle API
## As it allows for anonymous downloads without needing to sign in
from datasets import load_dataset # To load dataset without breaking ram
from multiprocessing import cpu_count # To get the number of cores
from sentence_transformers import SentenceTransformer # For embedding the text
import torch # For gpu 
import pandas as pd # Data manipulation
from time import time, sleep # Track time taken
import os # Folder and file creation
from tqdm import tqdm # Progress bar
tqdm.pandas() # Progress bar for pandas
from mixedbread_ai.client import MixedbreadAI # For embedding the text
from dotenv import dotenv_values # To load environment variables
import numpy as np # For array manipulation
from huggingface_hub import HfApi # To transact with huggingface.co
from glob import glob # To get all files in a folder
from datetime import datetime # To get the current date and time
import kagglehub # To download the dataset from Kaggle

# Track time
start = time()

################################################################################

# Flag to force download and conversion even if files already exist
FORCE = False

# Flag to embed the data locally, otherwise it will use mxbai api to embed
LOCAL = True

# Flag to upload the data to the Hugging Face Hub
UPLOAD = True

# Model to use for embedding
model_name = "mixedbread-ai/mxbai-embed-large-v1"

# Number of cores to use for multiprocessing
num_cores = cpu_count()-1

# Setup transaction details
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"
repo_type = "dataset"

# Subfolder in the repo of the dataset where the files will be stored
folder_in_repo = "data"

# Import secrets
config = dotenv_values(".env")

# Print configuration
print(f"Configuration:")
print(f"FORCE: {FORCE}")
print(f"LOCAL: {LOCAL}")
print(f"UPLOAD: {UPLOAD}")

################################################################################
# Download the dataset

# Dataset name
dataset_path = 'Cornell-University/arxiv'

# Download folder
download_folder = kagglehub.dataset_download(dataset_path, force_download=FORCE)

# Data file path
download_file = f'{download_folder}/arxiv-metadata-oai-snapshot.json'

################################################################################
# Convert to parquet

# https://huggingface.co/docs/datasets/en/about_arrow#memory-mapping
# Load metadata
print(f"Loading json metadata")
dataset = load_dataset("json", data_files= str(f"{download_file}"), num_proc=num_cores)

# Split metadata by year
# Convert to pandas
print(f"Loading metadata into pandas")
arxiv_metadata_all = dataset['train'].to_pandas()

# Create folder
split_folder = f"{download_file.replace('.json','-split')}"
os.makedirs(split_folder, exist_ok=True)

########################################

# Function to extract year from arxiv id
# https://info.arxiv.org/help/arxiv_identifier.html
# Function to extract Month and year of publication using arxiv ID
def extract_month_year(arxiv_id, what='month'):
    # Identify the relevant YYMM part based on the arXiv ID format
    yymm = arxiv_id.split('/')[-1][:4] if '/' in arxiv_id else arxiv_id.split('.')[0]
    
    # Convert the year-month string to a datetime object
    date = datetime.strptime(yymm, '%y%m')
    
    # Return the desired part based on the input parameter
    return date.strftime('%B') if what == 'month' else int(date.strftime('%Y'))
    
########################################

# Extract the year from the arxiv id column
arxiv_metadata_all['year'] =  arxiv_metadata_all['id'].progress_apply(extract_month_year, what='year')

# Group by the year and save each group as a separate Parquet file
for year, group in arxiv_metadata_all.groupby('year'):

    # Parquet file name
    split_file = f'{split_folder}/{year}.parquet'

    # Check if parquet file exists
    if not os.path.exists(split_file) or FORCE:

        print(f'Saving {split_file}.parquet')

        # Save each group to a separate Parquet file
        group.to_parquet(split_file, index=False)

    else:

        print(f'{split_file}.parquet already exists')
        print('Skipping')

        continue

################################################################################

# Gather split files
split_files = glob(f'{split_folder}/*.parquet')
print(f"Found {len(split_files)} split files")

# Create folder
embed_folder = f"{split_folder}-embed"
os.makedirs(embed_folder, exist_ok=True)

################################################################################

# Load Model

if LOCAL:
    # Make the app device agnostic
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load a pretrained Sentence Transformer model and move it to the appropriate device
    print(f"Loading model {model_name} to device: {device}")
    model = SentenceTransformer(model_name)
    model = model.to(device)
else:
    print("Setting up mxbai client")
    # Setup mxbai
    mxbai_api_key = config["MXBAI_API_KEY"]
    mxbai = MixedbreadAI(api_key=mxbai_api_key)

# Function that does the embedding
def embed(input_text):
    
    if LOCAL:

        # Calculate embeddings by calling model.encode(), specifying the device
        embedding = model.encode(input_text, device=device, precision="float32")

        # Enforce 32-bit float precision
        embedding = np.array(embedding, dtype=np.float32)

    else:

        # Sleep to avoid rate limit
        sleep(0.2)

        # Calculate embeddings by calling mxbai.embeddings()
        result = mxbai.embeddings(
        model='mixedbread-ai/mxbai-embed-large-v1',
        input=input_text,
        normalized=True,
        encoding_format='float',
        truncation_strategy='end'
        )

        # Enforce 32-bit float precision
        embedding = np.array(result.data[0].embedding, dtype=np.float32)

    return embedding

################################################################################

# Loop through each split file
for split_file in split_files:

    print('#'*80)

    # Load metadata
    print(f"Loading metadata file: {split_file}")   
    arxiv_metadata_split = pd.read_parquet(split_file)

    # Drop duplicates based on the 'id' column
    arxiv_metadata_split = arxiv_metadata_split.drop_duplicates(subset='id', keep='last', ignore_index=True)

    # Create a column for embeddings
    print(f"Creating embeddings for: {len(arxiv_metadata_split)} entries")
    arxiv_metadata_split["vector"] = arxiv_metadata_split["abstract"].progress_apply(embed)

####################
    print("Adding url and month columns")

    # Add URL column
    arxiv_metadata_split['url'] = 'https://arxiv.org/abs/' + arxiv_metadata_split['id']

    # Add month column
    arxiv_metadata_split['month'] = arxiv_metadata_split['id'].progress_apply(extract_month_year, what='month')

####################
    print("Removing newline characters from title, authors, categories, abstract")

    # Remove newline characters from authors, title, abstract and categories columns
    arxiv_metadata_split['title'] = arxiv_metadata_split['title'].astype(str).str.replace('\n', ' ', regex=False)

    arxiv_metadata_split['authors'] = arxiv_metadata_split['authors'].astype(str).str.replace('\n', ' ', regex=False)

    arxiv_metadata_split['categories'] = arxiv_metadata_split['categories'].astype(str).str.replace('\n', ' ', regex=False)

    arxiv_metadata_split['abstract'] = arxiv_metadata_split['abstract'].astype(str).str.replace('\n', ' ', regex=False)

####################
    print("Trimming title, authors, categories, abstract")

    # Trim title to 512 characters
    arxiv_metadata_split['title'] = arxiv_metadata_split['title'].progress_apply(lambda x: x[:508] + '...' if len(x) > 512 else x)

    # Trim categories to 128 characters
    arxiv_metadata_split['categories'] = arxiv_metadata_split['categories'].progress_apply(lambda x: x[:124] + '...' if len(x) > 128 else x)

    # Trim authors to 128 characters
    arxiv_metadata_split['authors'] = arxiv_metadata_split['authors'].progress_apply(lambda x: x[:124] + '...' if len(x) > 128 else x)

    # Trim abstract to 3072 characters
    arxiv_metadata_split['abstract'] = arxiv_metadata_split['abstract'].progress_apply(lambda x: x[:3068] + '...' if len(x) > 3072 else x)

####################
    # Selecting id, vector and $meta to retain
    selected_columns = ['id', 'vector', 'title', 'abstract', 'authors', 'categories', 'month', 'year', 'url']

    # Save the embedded file
    embed_filename = f'{embed_folder}/{os.path.basename(split_file)}'
    print(f"Saving embedded dataframe to: {embed_filename}")
    # Keeping index=False to avoid saving the index column as a separate column in the parquet file
    # This keeps milvus from throwing an error when importing the parquet file
    arxiv_metadata_split[selected_columns].to_parquet(embed_filename, index=False)

################################################################################

# Upload the new embeddings to the repo
if UPLOAD:

    print(f"Uploading all embeddings to: {repo_id}")
    access_token =  config["HF_API_KEY"]
    api = HfApi(token=access_token)

    # Upload all files within the folder to the specified repository
    api.upload_folder(repo_id=repo_id, folder_path=embed_folder, path_in_repo=folder_in_repo, repo_type="dataset")
################################################################################

# Track time
end = time()

print('#'*80)
print(f"It took a total of: {(end - start)/3600} hours")
print("Done!")