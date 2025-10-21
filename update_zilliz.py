# Connect using a MilvusClient object
from pymilvus.bulk_writer import bulk_import
from pymilvus.bulk_writer.stage_manager import StageManager
from pymilvus.bulk_writer.stage_file_manager import StageFileManager
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient, DataType
from huggingface_hub import snapshot_download
from glob import glob
from tqdm import tqdm
from datetime import datetime
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
print("Initiating MilvusClient, StageManager, and StageFileManager")
# Initialize a MilvusClient instance
# Replace uri and token with your own
client = MilvusClient(
    uri=CLUSTER_ENDPOINT, # Cluster endpoint obtained from the console
    token=ZILLIZ_TOKEN # API key or a colon-separated cluster username and password
)

stage_manager = StageManager(
    cloud_endpoint=BASE_URL,
    api_key=ZILLIZ_API_KEY
)

stage_file_manager = StageFileManager(
    cloud_endpoint=BASE_URL,
    api_key=ZILLIZ_API_KEY,
    stage_name=STAGE_NAME,
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

current_year = datetime.now().year

repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"
repo_type = "dataset"
local_dir = "volumes/milvus"
allow_patterns = "*.parquet"

dataset_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)
print(f"Dataset downloaded at {dataset_dir}")

dataset_files = glob(f"{dataset_dir}/data/*.parquet")
dataset_files.sort()

# Get the update file
update_file = f"{dataset_dir}/data/{current_year}.parquet"
print(f"Using {update_file} for update")
print('*'*80)
################################################################################
# Setup zilliz stage

# Upload data to stage. https://docs.zilliz.com/docs/manage-stages#upload-data-into-a-stage
    
print(f"Uploading: {update_file}")
upload_result = stage_file_manager.upload_file_to_stage(
    source_file_path=update_file, 
    target_stage_path=STAGE_PATH
    )
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

