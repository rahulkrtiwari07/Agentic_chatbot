from pymongo import MongoClient, errors
import pprint
import os
import sys

from dotenv import load_dotenv
from mongooperations import get_mongo_client

# Load environment variables
load_dotenv('.env.example')

# Constants
DATABASE_NAME = "chat_db"
MONGO_URI = os.getenv('MONGO_URL')

def retrieve_data(client, db_name, session_id=None, collection_name=None):
    """
    Retrieves documents from a specific collection (or all collections if not specified)
    in the specified MongoDB database. Optionally filters by session_id.

    Returns a dictionary with collection names as keys and lists of documents as values.
    """
    try:
        db = client[db_name]
        all_data = {}
        query = {"session_id": session_id} if session_id else {}

        # Only retrieve from specified collection
        if collection_name:
            if collection_name in db.list_collection_names():
                collection = db[collection_name]
                documents = list(collection.find(query, {"_id": 0}))
                all_data[collection_name] = documents
            else:
                print(f"Collection '{collection_name}' not found in database.")
        else:
            # Retrieve from all collections
            for name in db.list_collection_names():
                collection = db[name]
                documents = list(collection.find(query, {"_id": 0}))
                all_data[name] = documents

        return all_data

    except errors.PyMongoError as e:
        print(f"MongoDB Error: {e}")
        return {}

def main():
    if not MONGO_URI:
        print("Environment variable 'MONGO_URL' is not set.")
        sys.exit(1)

    try:
        mongo_client = get_mongo_client(MONGO_URI)

        # Example usage with both optional filters
        collection_name = "user_data_1"

        data = retrieve_data(mongo_client, DATABASE_NAME, collection_name=collection_name)
        pprint.pprint(data)

    except errors.ConnectionFailure as ce:
        print(f"Connection Error: {ce}")
        sys.exit(1)

if __name__ == "__main__":
    main()
