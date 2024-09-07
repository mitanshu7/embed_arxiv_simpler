## Download the arXiv metadata from Kaggle
## https://www.kaggle.com/datasets/Cornell-University/arxiv

## Requires the Kaggle API to be installed
## Using subprocess to run the Kaggle CLI commands instead of Kaggle API
## As it allows for anonymous downloads without needing to sign in
import subprocess
import os

################################################################################

## Defining paths
# Dataset name
dataset_path = 'Cornell-University/arxiv'
# Download folder
download_folder = 'data'
# Data file path
data_file_path = f'{download_folder}/arxiv-metadata-oai-snapshot.json'

## Download the dataset if it doesn't exist
if not os.path.exists(data_file_path):
    print(f'Downloading {data_file_path}')
    subprocess.run(['kaggle', 'datasets', 'download', '--dataset', dataset_path, '--path', download_folder, '--unzip'])
    print(f'Downloaded {data_file_path}')
else:
    print(f'{data_file_path} already exists')
    print('Skipping download')

## Done
print('Done!')