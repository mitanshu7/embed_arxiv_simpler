# File description:
# This script merges two dataframes, one containing embeddings and the other containing metadata, on the 'id' column.
# It then creates a new column '$meta' which is a json string of the metadata.
# Finally, it drops the unnecessary columns and saves the merged dataframe to a parquet file.
# The output file is compatible with Milvus, a vector database.

# Import required libraries
import pandas as pd
import json
from glob import glob
import os

# Declare filenames
embeddings_folder = 'embeddings_data/data'
metadata_folder = 'data/arxiv-metadata-oai-snapshot-split'    
output_folder = 'embeddings_data/data_with_metadata'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all parquet files in embeddings folder
embeddings_files = glob(embeddings_folder + '/*.parquet')
embeddings_files.sort()

# Get list of all parquet files in metadata folder
metadata_files = glob(metadata_folder + '/*.parquet')
metadata_files.sort()

# Loop through each pair of embeddings and metadata files
for embeddings_file, metadata_file in zip(embeddings_files, metadata_files):


    # Load dataframes
    embeddings = pd.read_parquet(embeddings_file)
    metadata = pd.read_parquet(metadata_file)

    # Rename columns
    metadata.rename(columns={'title': 'Title', 'authors': 'Authors', 'abstract': 'Abstract'}, inplace=True)

    # Add URL column
    metadata['URL'] = 'https://arxiv.org/abs/' + metadata['id']

    # Merge dataframes on 'id'
    merged = pd.merge(embeddings, metadata, on='id')

    # Create milvus compatible parquet file
    # It has 3 columns, id, vector, $meta, where
    # id is the paper_id, vector is the embedding, and $meta is a json string of the metadata
    merged['$meta'] = merged[['Title', 'Authors', 'Abstract', 'URL']].apply(lambda row: json.dumps(row.to_dict()), axis=1)

    # Keep neccessary columns
    merged = merged[['id', 'vector', '$meta']]

    # Save to parquet
    output_file = os.path.join(output_folder, os.path.basename(embeddings_file))
    merged.to_parquet(output_file, index=False)

