# Connect using a MilvusClient object
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
import os

################################################################################
# Configuration
load_dotenv(".env")

CLUSTER_ENDPOINT = os.getenv('CLUSTER_ENDPOINT')
TOKEN = os.getenv('TOKEN')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# Initialize a MilvusClient instance
# Replace uri and token with your own
client = MilvusClient(
    uri=CLUSTER_ENDPOINT, # Cluster endpoint obtained from the console
    token=TOKEN # API key or a colon-separated cluster username and password
)

################################################################################

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

################################################################################
# Create index

# Set up the index parameters
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
        field_name="vector",
        metric_type="HAMMING",
        index_type="BIN_IVF_FLAT",
        index_name="vector_index",
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
    index_name="vector_index"
)

print(res)

################################################################################

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
