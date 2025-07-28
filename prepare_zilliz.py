# Connect using a MilvusClient object
from pymilvus.bulk_writer import bulk_import
import requests
from dotenv import load_dotenv
import os
from pymilvus.stage.stage_operation import StageOperation
from pymilvus import MilvusClient, DataType
from huggingface_hub import snapshot_download
from glob import glob
from tqdm import tqdm
################################################################################
# Configuration
load_dotenv(".env")

CLUSTER_ENDPOINT = os.getenv('CLUSTER_ENDPOINT')
ZILLIZ_TOKEN = os.getenv('ZILLIZ_TOKEN')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', "default_collection")
CLUSTER_ID = os.getenv('CLUSTER_ID')
ZILLIZ_API_KEY = os.getenv('ZILLIZ_API_KEY')
STAGE_NAME= os.getenv('STAGE_NAME', "default_stage")
STAGE_PATH= os.getenv('STAGE_PATH',"default_stage_path")
PROJECT_ID = os.getenv('PROJECT_ID')
CLOUD_REGION = os.getenv('CLOUD_REGION')
BASE_URL = os.getenv('BASE_URL')
INDEX_NAME = os.getenv('INDEX_NAME', 'default_index')

# Print configuration
print('='*80)
print("Configuration:")
print(f"CLUSTER_ENDPOINT {CLUSTER_ENDPOINT}")
print(f"ZILLIZ_TOKEN {ZILLIZ_TOKEN}")
print(f"COLLECTION_NAME {COLLECTION_NAME}")
print(f"CLUSTER_ID {CLUSTER_ID}")
print(f"ZILLIZ_API_KEY {ZILLIZ_API_KEY}")
print(f"STAGE_NAME {STAGE_NAME}")
print(f"STAGE_PATH {STAGE_PATH}")
print(f"PROJECT_ID {PROJECT_ID}")
print(f"CLOUD_REGION {CLOUD_REGION}")
print(f"BASE_URL {BASE_URL}")
print('='*80)
################################################################################
# Reset zilliz
print("!"*80)

# Initialize a MilvusClient instance
# Replace uri and token with your own
client = MilvusClient(
    uri=CLUSTER_ENDPOINT, # Cluster endpoint obtained from the console
    token=ZILLIZ_TOKEN # API key or a colon-separated cluster username and password
)

########################################

# Drop any of the pre-existing collections
# Need to drop it because otherwise milvus does not check for (and keeps)
# duplicate records
print(f"Dropping collection: {COLLECTION_NAME}")
client.drop_collection(
    collection_name=COLLECTION_NAME
)

# Dataset schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False
)

# Add the fields to the schema
schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=32, is_primary=True)

schema.add_field(field_name="vector", datatype=DataType.BINARY_VECTOR, dim=1024)

schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="authors", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=3072)
schema.add_field(field_name="categories", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="month", datatype=DataType.VARCHAR, max_length=16)
schema.add_field(field_name="year", datatype=DataType.INT64, max_length=8, is_clustering_key=True)
schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=64)

print("Issues with scheme: ", schema.verify())

# Create a collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema
)

########################################
# Create index

# Set up the index parameters
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
        field_name="vector",
        metric_type="HAMMING",
        index_type="BIN_IVF_FLAT",
        index_name=INDEX_NAME,
        params={ "nlist": 128 }
    )

print("Creating Index file.")

# Create an index file
res = client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True # Wait for index creation to complete before returning. 
)

print(res)

print("Listing indexes.")

# List indexes
res = client.list_indexes(
    collection_name=COLLECTION_NAME
)

print(res)

print("Describing Index.")

# Describe index
res = client.describe_index(
    collection_name=COLLECTION_NAME,
    index_name=INDEX_NAME
)

print(res)

########################################

# Load the collection

print(f"Loading Collection: {COLLECTION_NAME}")

client.load_collection(
    collection_name=COLLECTION_NAME,
    replica_number=1 # Number of replicas to create on query nodes. 
)

res = client.get_load_state(
    collection_name=COLLECTION_NAME
)

print("Collection load state:")
print(res)

print("!"*80)
################################################################################
# Download the dataset
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"
repo_type = "dataset"
local_dir = "volumes/milvus"
allow_patterns = "*.parquet"

dataset_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)
print(f"Dataset downloaded at {dataset_dir}")

# print(f"Modifying dataset directory from '{dataset_dir}' to '{dataset_dir}/data' to only upload parquet files and not the 'data' folder")
# dataset_dir = f"{dataset_dir}/data"

dataset_files = glob(f"{dataset_dir}/data/*.parquet")
dataset_files.sort()
print(f"Found {len(dataset_files)} files!")
print('*'*80)
################################################################################
# Setup zilliz stage
########################################
# Create a stage. https://docs.zilliz.com/docs/manage-stages#create-a-stage
def create_stage():
    
    headers = {'Authorization': f'Bearer {ZILLIZ_API_KEY}',
                'Content-Type': 'application/json'}
    
    data = {
        "projectId": PROJECT_ID,
        "regionId": CLOUD_REGION,
        "stageName": STAGE_NAME
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v2/stages/create", headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
        return response.json()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
        
    except Exception as err:
        print(f"Unexpected error: {err}")

# Create a stage 
print(f"Creating stage: {STAGE_NAME}")
create_stage_result = create_stage()
print(create_stage_result)
print('*'*80)
########################################

# Upload data to stage. https://docs.zilliz.com/docs/manage-stages#upload-data-into-a-stage
def upload_to_stage(local_dir_or_file_path:str):
    
    stage_operation = StageOperation(
        cloud_endpoint=BASE_URL,
        api_key=ZILLIZ_API_KEY,
        stage_name=STAGE_NAME,
        path=STAGE_PATH
    )
    
    result = stage_operation.upload_file_to_stage(local_dir_or_file_path)
    
    return result

# Upload to stage
for dataset_file in tqdm(dataset_files, desc="Uploading"):
    
    print(f"Uploading: {dataset_file}")
    upload_result = upload_to_stage(dataset_file)
    print(upload_result)
    
print('*'*80)
########################################

# Import data into collection via stage. https://docs.zilliz.com/docs/import-data-via-sdks#import-data-via-stage
def import_from_stage(data_file:str):

    response = bulk_import(
        url=BASE_URL,
        api_key=ZILLIZ_API_KEY,
        cluster_id=CLUSTER_ID,
        collection_name=COLLECTION_NAME,
        stage_name=STAGE_NAME,
        data_paths=[[f"{STAGE_PATH}{data_file}"]], # Dont add a '/' in between since STAGE_PATH already ends with '/'
        db_name='' # NEED to keep empty to avoid cluster collection not found error.
    )
    
    return response.json()
    
# Import to collection
print(f"Importing data from '{STAGE_NAME}/{STAGE_PATH}' to '{COLLECTION_NAME}'")

for dataset_file in tqdm(dataset_files, desc="Importing"):
    
    # Get filename for reference
    dataset_file = os.path.basename(dataset_file)
    
    print(f"Importing: {dataset_file}")
    
    # import a single file
    import_result = import_from_stage(dataset_file)
    
    print(import_result)
    
########################################

# Delete a stage. https://docs.zilliz.com/docs/manage-stages#delete-a-stage
print("!"*80)
def delete_stage():
    
    headers = {'Authorization': f'Bearer {ZILLIZ_API_KEY}',
                'Content-Type': 'application/json'}
    try:
        response = requests.delete(f"{BASE_URL}/v2/stages/{STAGE_NAME}", headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
        return response.json()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
        
    except Exception as err:
        print(f"Unexpected error: {err}")

# Create a stage 
print(f"Deleting stage: {STAGE_NAME}")
delete_stage_result = delete_stage()
print(delete_stage_result)
print("!"*80)