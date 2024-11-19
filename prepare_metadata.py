## Download the arXiv metadata from Kaggle
## https://www.kaggle.com/datasets/Cornell-University/arxiv

## Requires the Kaggle API to be installed
## Using subprocess to run the Kaggle CLI commands instead of Kaggle API
## As it allows for anonymous downloads without needing to sign in

import subprocess
import os
import pandas as pd
from time import time

# Track time
start_time = time()

################################################################################

# Flag to force download and conversion even if files already exist
FORCE = False

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

    print(f'Downloading {download_file}')

    subprocess.run(['kaggle', 'datasets', 'download', '--dataset', dataset_path, '--path', download_folder, '--unzip'])
    
    print(f'Downloaded {download_file}')

else:

    print(f'{download_file} already exists')
    print('Skipping download')

################################################################################
# Convert to parquet

# Load metadata
arxiv_metadata_all = pd.read_json(download_file, lines=True, convert_dates=True)

# Parquet file name
parquet_file = f'{download_folder}/arxiv-metadata-oai-snapshot.parquet'

# Check if parquet file exists
if not os.path.exists(parquet_file) or FORCE:

    print(f'Converting {download_file} to parquet')

    # Save to parquet format
    arxiv_metadata_all.to_parquet(parquet_file, index=False)
    
    print(f'Saved {parquet_file}')

else:

    print(f'{parquet_file} already exists')
    print('Skipping conversion')

################################################################################
# Split metadata by year

# Create folder
split_folder = f"{parquet_file.replace('.parquet','-split')}"
os.makedirs(split_folder, exist_ok=True)

########################################

# Function to extract year from arxiv id
# https://info.arxiv.org/help/arxiv_identifier.html
def extract_year(arxiv_id):

    # Old scheme
    if '/' in arxiv_id:
        return arxiv_id.split('/')[1][:2]
    
    # New scheme
    else:
        return arxiv_id[:2]
    
########################################

# Extract the year from the arxiv id column
arxiv_metadata_all['year'] =  arxiv_metadata_all['id'].apply(extract_year)

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

# Track time
end_time = time()

# Print time taken
print(f'Time taken to prepare metadata: {(end_time - start_time)/60} minutes')