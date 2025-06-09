from pymongo import MongoClient, errors
import pprint
import os
import sys

from dotenv import load_dotenv
from mongooperations import get_mongo_client

load_dotenv('.env.example')

# Get Mongo URI from environment variable
mongo_uri = os.getenv('MONGO_URL')
if not mongo_uri:
    print("Environment variable 'MONGO_URL' is not set.")
    sys.exit(1)


# Database name
database_name = "chat_db"

def retrieve_all_data_from_db(client, db_name):
    """
    Retrieves all documents from all collections in the specified MongoDB database.
    Returns a dictionary with collection names as keys and lists of documents as values.
    """
    try:
        db = client[db_name]
        all_data = {}

        # List all collections in the database
        collection_names = db.list_collection_names()

        for collection_name in collection_names:
            collection = db[collection_name]
            documents = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
            all_data[collection_name] = documents

        return all_data

    except errors.PyMongoError as e:
        print(f"MongoDB Error: {e}")
        return {}

# Initialize client and retrieve data
try:
    client = get_mongo_client(mongo_uri)
    data = retrieve_all_data_from_db(client, database_name)
    pprint.pprint(data)
except errors.ConnectionError as ce:
    print(f"Connection Error: {ce}")
    sys.exit(1)
