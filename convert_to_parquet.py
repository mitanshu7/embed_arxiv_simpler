# Import required library
import pandas as pd

################################################################################

# Metadata file
data_folder = 'data'
data_file = f"{data_folder}/arxiv-metadata-oai-snapshot.json"

# Load metadata
arxiv_metadata_all = pd.read_json(data_file, lines=True, convert_dates=True)

# Save to parquet format
arxiv_metadata_all.to_parquet(f'{data_folder}/arxiv-metadata-oai-snapshot.parquet', index=False)