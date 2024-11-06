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

# Year to diff merge
year = '24'

# Declare filenames
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"
old_embeddings_file = f'{repo_id}/data/{year}.parquet'

diff_embeddings_file = f'data/arxiv-metadata-oai-snapshot-trim-split-diff-embed/{year}.parquet'

metadata_file = f'data/arxiv-metadata-oai-snapshot-split/{year}.parquet'    

output_folder = 'embeddings_data/data_with_metadata'
os.makedirs(output_folder, exist_ok=True)

output_file = f'{output_folder}/{year}.parquet'

# Load dataframes
embeddings = pd.concat([pd.read_parquet(old_embeddings_file), pd.read_parquet(diff_embeddings_file)])
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
merged.to_parquet(output_file, index=False)

