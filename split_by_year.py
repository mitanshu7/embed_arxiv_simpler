# Import required library
import pandas as pd
import os

################################################################################

# Metadata file
data_folder = 'data'
data_file = f"{data_folder}/arxiv-metadata-oai-snapshot-trimmed.parquet"

# Create folder
split_folder = f"{data_file.replace('.parquet','')}-split"
os.makedirs(split_folder, exist_ok=True)

# Function to extract year from arxiv id
# https://info.arxiv.org/help/arxiv_identifier.html
def extract_year(arxiv_id):
    # Old scheme
    if '/' in arxiv_id:
        return arxiv_id.split('/')[1][:2]
    # New scheme
    else:
        return arxiv_id[:2]

################################################################################

# Load metadata
arxiv_metadata_all = pd.read_parquet(data_file)

# Extract the year from the arxiv id column
arxiv_metadata_all['year'] =  arxiv_metadata_all['id'].apply(extract_year)

# Step 2: Group by the year and save each group as a separate Parquet file
for year, group in arxiv_metadata_all.groupby('year'):

    # Save each group to a separate Parquet file
    group.to_parquet(f'{split_folder}/{year}.parquet', index=False)