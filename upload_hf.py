# Import required libraries
from huggingface_hub import HfApi # To transact with huggingface.co
from dotenv import dotenv_values # get secrets

# Setup the Hugging Face API
# Import environment variables
config = dotenv_values(".env")
access_token =  config["HF_API_KEY"]
api = HfApi(token=access_token)

# Verify the API
user = api.whoami()
print(user)

# Setup transaction details
## CHANGE BELOW LINE ##
folder_path = "data/arxiv-metadata-oai-snapshot-trim-split-embed" # Which folder to upload
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"  # Which repository to upload to
subfolder = "data"  # Optional subfolder within the repository

# Upload all files within the folder to the specified repository
api.upload_folder(repo_id=repo_id, folder_path=folder_path, path_in_repo=subfolder, repo_type="dataset")
