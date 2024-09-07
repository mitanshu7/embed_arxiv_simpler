# Import required library
import pandas as pd

################################################################################

# Metadata file
data_folder = "data"
data_file = f"{data_folder}/arxiv-metadata-oai-snapshot.parquet"

# Load metadata
arxiv_metadata_all = pd.read_parquet(data_file)

# Selecting id and abstract to retain
selected_columns = ['id', 'abstract']

# Save to a Parquet file with only the selected columns
arxiv_metadata_all[selected_columns].to_parquet(f"{data_folder}/arxiv-metadata-oai-snapshot-trim.parquet", index=False)