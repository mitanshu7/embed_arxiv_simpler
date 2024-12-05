## Download the arXiv metadata from Kaggle
## https://www.kaggle.com/datasets/Cornell-University/arxiv

## Requires the Kaggle API to be installed
## Using subprocess to run the Kaggle CLI commands instead of Kaggle API
## As it allows for anonymous downloads without needing to sign in
import subprocess
from datasets import load_dataset # To load dataset without breaking ram
from multiprocessing import cpu_count # To get the number of cores
from sentence_transformers import SentenceTransformer # For embedding the text
import torch # For gpu 
import pandas as pd # Data manipulation
from huggingface_hub import snapshot_download # Download previous embeddings
import json # To make milvus compatible $meta
import os # Folder and file creation
from tqdm import tqdm # Progress bar
tqdm.pandas() # Progress bar for pandas
from mixedbread_ai.client import MixedbreadAI # For embedding the text
from dotenv import dotenv_values # To load environment variables
import numpy as np # For array manipulation
from huggingface_hub import HfApi # To transact with huggingface.co
import sys # To quit the script
import datetime # get current year
from time import time, sleep # To time the script
from datetime import datetime # To get the current date and time

# Start timer
start = time()

################################################################################
# Configuration

# Year to update embeddings for, get and set the current year
year = str(datetime.datetime.now().year)[2:]

# Flag to force download and conversion even if files already exist
FORCE = True

# Flag to embed the data locally, otherwise it will use mxbai api to embed
LOCAL = False

# Flag to upload the data to the Hugging Face Hub
UPLOAD = True

# Flag to binarise the data
BINARY = True

# Flag to BMRL the data
BMRL = True

########################################

# Model to use for embedding
model_name = "mixedbread-ai/mxbai-embed-large-v1"

# Number of cores to use for multiprocessing
num_cores = cpu_count()-1

# Setup transaction details
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

# Import secrets
config = dotenv_values(".env")

################################################################################
# Download the dataset

# Dataset name
dataset_path = 'Cornell-University/arxiv'

# Download folder
download_folder = 'data'

# Data file path
download_file = f'{download_folder}/arxiv-metadata-oai-snapshot.json'

## Download the dataset if it doesn't exist
if not os.path.exists(download_file) or FORCE:

    print(f'Downloading {download_file}, if it exists it will be overwritten')
    print('Set FORCE to False to skip download if file already exists')

    subprocess.run(['kaggle', 'datasets', 'download', '--dataset', dataset_path, '--path', download_folder, '--unzip'])
    
    print(f'Downloaded {download_file}')

else:

    print(f'{download_file} already exists, skipping download')
    print('Set FORCE = True to force download')

################################################################################
# Filter by year and convert to parquet

# https://huggingface.co/docs/datasets/en/about_arrow#memory-mapping
# Load metadata
print(f"Loading json metadata")
dataset = load_dataset("json", data_files= str(f"{download_file}"))

# Split metadata by year
# Convert to pandas
print(f"Converting metadata into pandas")
arxiv_metadata_all = dataset['train'].to_pandas()

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
    return date.strftime('%B') if what == 'month' else date.strftime('%Y')
########################################

# Add year to metadata
print(f"Adding year to metadata")
arxiv_metadata_all['year'] =  arxiv_metadata_all['id'].progress_apply(extract_month_year, what='year')

# Filter by year
print(f"Filtering metadata by year: {year}")
arxiv_metadata_split = arxiv_metadata_all[arxiv_metadata_all['year'] == year]

################################################################################
# Load Model

if LOCAL:

    print(f"Setting up local embedding model")
    print("To use mxbai API, set LOCAL = False")

    # Make the app device agnostic
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load a pretrained Sentence Transformer model and move it to the appropriate device
    print(f"Loading model {model_name} to device: {device}")
    model = SentenceTransformer(model_name)
    model = model.to(device)
else:
    print("Setting up mxbai API client")
    print("To use local resources, set LOCAL = True")
    # Setup mxbai
    mxbai_api_key = config["MXBAI_API_KEY"]
    mxbai = MixedbreadAI(api_key=mxbai_api_key)

########################################
# Function that does the embedding
def embed(input_text):
    
    if LOCAL:

        # Calculate embeddings by calling model.encode(), specifying the device
        embedding = model.encode(input_text, device=device, precision="float32")

        # Enforce 32-bit float precision
        embedding = np.array(embedding, dtype=np.float32)

    else:
        
        # Avoid rate limit from api
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
########################################

################################################################################
# Gather preexisting embeddings

# Subfolder in the repo of the dataset where the file is stored
folder_in_repo = "data"
allow_patterns = f"{folder_in_repo}/{year}.parquet"

# Where to store the local copy of the dataset
local_dir = repo_id

# Set repo type
repo_type = "dataset"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Download the repo
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)

try:

    # Gather previous embed file
    previous_embed = f'{local_dir}/{folder_in_repo}/{year}.parquet'

    # Load previous_embed
    print(f"Loading previously embedded file: {previous_embed}")   
    previous_embeddings = pd.read_parquet(previous_embed)

except Exception as e:
    print(f"Errored out with: {e}")
    print(f"No previous embeddings found for year: {year}")
    print("Creating new embeddings for all papers")
    previous_embeddings = pd.DataFrame(columns=['id', 'vector', '$meta'])

########################################
# Embed the new abstracts

# Find papers that are not in the previous embeddings
new_papers = arxiv_metadata_split[~arxiv_metadata_split['id'].isin(previous_embeddings['id'])]

# Drop duplicates based on the 'id' column
new_papers = new_papers.drop_duplicates(subset='id', keep='last', ignore_index=True)

# Number of new papers
num_new_papers = len(new_papers)

# What if there are no new papers?
if num_new_papers == 0:
    print(f"No new papers found for year: {year}")
    print("Exiting")
    sys.exit()

# Create a column for embeddings
print(f"Creating new embeddings for: {num_new_papers} entries")
new_papers["vector"] = new_papers["abstract"].progress_apply(embed)

####################
# Add URL column
arxiv_metadata_split['url'] = 'https://arxiv.org/abs/' + arxiv_metadata_split['id']

# Add month column
arxiv_metadata_split['month'] = arxiv_metadata_split['id'].progress_apply(extract_month_year, what='month')

####################
# Remove newline characters from authors, title, abstract and categories columns
arxiv_metadata_split['title'] = arxiv_metadata_split['title'].astype(str).str.replace('\n', ' ', regex=False)

arxiv_metadata_split['authors'] = arxiv_metadata_split['authors'].astype(str).str.replace('\n', ' ', regex=False)

arxiv_metadata_split['categories'] = arxiv_metadata_split['categories'].astype(str).str.replace('\n', ' ', regex=False)

arxiv_metadata_split['abstract'] = arxiv_metadata_split['abstract'].astype(str).str.replace('\n', ' ', regex=False)

####################
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

# Merge previous embeddings and new embeddings
new_embeddings = pd.concat([previous_embeddings, new_papers[selected_columns]])

# Create embed folder
embed_folder = f"{year}-diff-embed"
os.makedirs(embed_folder, exist_ok=True)

# Save the embedded file
embed_filename = f'{embed_folder}/{year}.parquet'
print(f"Saving newly embedded dataframe to: {embed_filename}")
# Keeping index=False to avoid saving the index column as a separate column in the parquet file
# This keeps milvus from throwing an error when importing the parquet file
new_embeddings.to_parquet(embed_filename, index=False)

################################################################################

# Upload the new embeddings to the repo
if UPLOAD:

    print(f"Uploading new embeddings to: {repo_id}")
    access_token =  config["HF_API_KEY"]
    api = HfApi(token=access_token)

    # Upload all files within the folder to the specified repository
    api.upload_folder(repo_id=repo_id, folder_path=embed_folder, path_in_repo=folder_in_repo, repo_type="dataset")

    print(f"Upload complete for year: {year}")

else:
    print("Not uploading new embeddings to the repo")
    print("To upload new embeddings, set UPLOAD to True")
################################################################################

# Binarise the data
if BINARY:

    print(f"Binarising the data for year: {year}")
    print("Set BINARY = False to not binarise the embeddings")

    # Function to convert dense vector to binary vector
    def dense_to_binary(dense_vector):
        return np.packbits(np.where(dense_vector >= 0, 1, 0))

    # Create a folder to store binary embeddings
    binary_folder = f"{year}-binary-embed"
    os.makedirs(binary_folder, exist_ok=True)

    # Convert the dense vectors to binary vectors
    new_embeddings['vector'] = new_embeddings['vector'].progress_apply(dense_to_binary)

    # Save the binary embeddings to a parquet file
    new_embeddings.to_parquet(f'{binary_folder}/{year}.parquet', index=False)

if BINARY and UPLOAD:

    # Setup transaction details
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"
    repo_type = "dataset"

    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # Subfolder in the repo of the dataset where the file is stored
    folder_in_repo = "data"

    print(f"Uploading binary embeddings to {repo_id} from folder {binary_folder}")

    # Upload all files within the folder to the specified repository
    api.upload_folder(repo_id=repo_id, folder_path=binary_folder, path_in_repo=folder_in_repo, repo_type=repo_type)

    print("Upload complete")

else:
    print("Not uploading Binary embeddings to the repo")
    print("To upload embeddings, set UPLOAD and BINARY both to True")


################################################################################

# BMRL the data
if BMRL:
    print(f"BMRL'ing the data for year: {year}")
    print("Set BMRL = False to not binarise and MRL the embeddings")

    # Function to chop a binary vector to a specific size
    def binary_to_mrl(binary_vector, size=512):
        return np.packbits(np.unpackbits(binary_vector)[:size])

    # Create a folder to store binary embeddings
    bmrl_folder = f"{year}-bmrl-embed"
    os.makedirs(bmrl_folder, exist_ok=True)

    # Convert the dense vectors to binary vectors
    new_embeddings['vector'] = new_embeddings['vector'].progress_apply(binary_to_mrl)

    # Save the binary embeddings to a parquet file
    new_embeddings.to_parquet(f'{bmrl_folder}/{year}.parquet', index=False)

if BMRL and UPLOAD:

    # Setup transaction details
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_bmrl"
    repo_type = "dataset"

    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # Subfolder in the repo of the dataset where the file is stored
    folder_in_repo = "data"

    print(f"Uploading binary embeddings to {repo_id} from folder {bmrl_folder}")

    # Upload all files within the folder to the specified repository
    api.upload_folder(repo_id=repo_id, folder_path=bmrl_folder, path_in_repo=folder_in_repo, repo_type=repo_type)

    print("Upload complete")

else:
    print("Not uploading BMRL embeddings to the repo")
    print("To upload embeddings, set UPLOAD and BMRL both to True")

################################################################################

# Track time
end = time()

# Calculate and show time taken
print(f"Time taken: {end - start} seconds")

print("Done!")