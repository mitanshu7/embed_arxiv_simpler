import pandas as pd 
import numpy as np
from glob import glob
import os
from tqdm import tqdm
tqdm.pandas()

#######################################################################################

# Function to convert dense vector to binary vector
def dense_to_binary(dense_vector):
    return np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes()


# Gather fp32 files
floats = glob('data/arxiv-metadata-oai-snapshot-split-embed/*.parquet')
floats.sort()

# Create a folder to store binary embeddings
folder_name = 'binary_embeddings'
os.makedirs(folder_name, exist_ok=True)

# Convert and save each file
for file in floats:
    

    print(f"Processing file: {file}")

    df = pd.read_parquet(file)

    df['vector'] = df['vector'].progress_apply(dense_to_binary)
    
    df.to_parquet(f'{folder_name}/{os.path.basename(file)}')

#######################################################################################

print("Conversion completed.")