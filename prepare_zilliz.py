# Connect using a MilvusClient object
from pymilvus.bulk_writer import bulk_import
import requests
from dotenv import load_dotenv
import os
from pymilvus.stage.stage_operation import StageOperation
################################################################################
# Configuration
load_dotenv(".env")

CLUSTER_ENDPOINT = os.getenv('CLUSTER_ENDPOINT')
TOKEN = os.getenv('TOKEN')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
CLUSTER_ID = os.getenv('CLUSTER_ID')
ZILLIZ_API_KEY = os.getenv('ZILLIZ_API_KEY')
STAGE_NAME= os.getenv('STAGE_NAME')
STAGE_PATH= os.getenv('STAGE_PATH')
PROJECT_ID = os.getenv('PROJECT_ID')
CLOUD_REGION = os.getenv('CLOUD_REGION')
BASE_URL = os.getenv('BASE_URL')

# Print configuration
print('='*80)
print("Configuration:")
print(f"CLUSTER_ENDPOINT {CLUSTER_ENDPOINT}")
print(f"TOKEN {TOKEN}")
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

# Import data into collection via stage. https://docs.zilliz.com/docs/import-data-via-sdks#import-data-via-stage
def import_from_stage():

    response = bulk_import(
        url=BASE_URL,
        api_key=ZILLIZ_API_KEY,
        cluster_id=CLUSTER_ID,
        collection_name=COLLECTION_NAME,
        stage_name=STAGE_NAME,
        data_paths=[[STAGE_PATH]],
        db_name='' # NEED TO KEEP EMPTY
    )
    
    return response.json()
    
if __name__ == '__main__':
    
    # Gather files
    file = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary/data/1991.parquet"
    
    # Create a stage 
    print(f"Creating stage: {STAGE_NAME}")
    stage_result = create_stage()
    print(stage_result)
    print('*'*80)

    
    # Upload to stage
    print(f"Uploading: {file}")
    upload_result = upload_to_stage(file)
    print(upload_result)
    print('*'*80)

    
    # Import to collection
    print(f"Importing data from '{STAGE_NAME}/{STAGE_PATH}' to '{COLLECTION_NAME}'")
    import_result = import_from_stage()
    print(import_result)
    print('*'*80)
