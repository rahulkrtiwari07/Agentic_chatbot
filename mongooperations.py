import datetime
import pymongo
from datetime import datetime, timedelta
import pymongo.errors
import bson

def get_mongo_client(uri):
    """Get a MongoDB client."""
    return pymongo.MongoClient(uri)

from datetime import datetime
import pymongo

def store_user_data(client, db_name, collection_name, session_id, chat_entry):
    """
    Stores a chat entry in MongoDB under the given session_id.
    If the document exists, appends to 'chat_history'; otherwise creates a new document.
    Each chat entry includes question, answer, and timestamp.
    """
    try:
        db = client[db_name]
        collection = db[collection_name]

        # Add timestamp to the chat entry
        chat_entry["timestamp"] = datetime.utcnow()

        chat_document = collection.find_one({"session_id": session_id})

        if chat_document:
            collection.update_one(
                {"_id": chat_document["_id"]},
                {"$push": {"chat_history": chat_entry}}
            )
        else:
            new_document = {
                "session_id": session_id,
                "chat_history": [chat_entry]
            }
            collection.insert_one(new_document)

        print(f"Chat entry stored for session: {session_id}")
        print(chat_entry)

    except (pymongo.errors.ServerSelectionTimeoutError,
            pymongo.errors.NetworkTimeout,
            pymongo.errors.ConnectionFailure) as e:
        print(f"Error connecting to MongoDB: {e}")


from datetime import datetime
import pymongo

def retrieve_user_data(client, db_name, collection_name, session_id, date=None):
    """
    Retrieves user data based on session_id and optional date.
    If session_id is 'admin_login', retrieves all chat logs.
    """
    try:
        db = client[db_name]
        collection = db[collection_name]

        chat_history = []

        if session_id == "admin_login":
            # Admin fetch: return all session documents with chat history
            for doc in collection.find({}, {"_id": 0, "session_id": 1, "chat_history": 1}):
                chat_history.append(doc)
            return chat_history

        else:
            match_stage = {"$match": {"session_id": session_id}}
            unwind_stage = {"$unwind": "$chat_history"}
            project_stage = {
                "$project": {
                    "_id": 0,
                    "timestamp": "$chat_history.timestamp",
                    "question": "$chat_history.question",
                    "answer": "$chat_history.answer"
                }
            }

            pipeline = [match_stage, unwind_stage, project_stage]

            if date:
                # Convert date string (DD-MM-YYYY) to datetime range for the entire day
                target_date = datetime.strptime(date, "%d-%m-%Y")
                next_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                date_filter = {
                    "$match": {
                        "timestamp": {
                            "$gte": target_date,
                            "$lt": next_day
                        }
                    }
                }
                pipeline.append(date_filter)

            for doc in collection.aggregate(pipeline):
                chat_history.append(doc)

            return chat_history

    except pymongo.errors.ConfigurationError as e:
        print(f"MongoDB Configuration Error: {e}")
    except (pymongo.errors.ServerSelectionTimeoutError,
            pymongo.errors.NetworkTimeout,
            pymongo.errors.ConnectionFailure) as e:
        print(f"MongoDB Connection Error: {e}")


def archive_data(client, db_name, source_collection_name, target_collection_name, days_threshold=30):
    """Transfers data from source collection to target collection if it's older than the threshold."""
    try:
        db = client[db_name]
        source_collection = db[source_collection_name]
        target_collection = db[target_collection_name]

        threshold_date = datetime.utcnow() - timedelta(days=days_threshold)
        delete_date = datetime.utcnow() - timedelta(days=180)

        query = {"date": {"$lt": threshold_date}}

        for document in source_collection.find(query):
            source_collection.delete_one({"_id": document["_id"]})
            target_collection.insert_one(document)
            
        query = {"date": {"$lt": delete_date}}

        for document in target_collection.find(query):
            target_collection.delete_one({"_id": document["_id"]})

        print(f"Successfully transferred documents older than {threshold_date} to archive collection.")
    except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.NetworkTimeout, pymongo.errors.ConnectionFailure) as e:
        print(f"Error connecting to MongoDB: {e}")

def get_unique_dates_by_email(client, db_name, collection_name, email):
  """
  Retrieves all unique date values for a given email from a MongoDB collection.

  Args:
    client: A pymongo MongoClient instance.
    db_name: The name of the database.
    collection_name: The name of the collection.
    email: The email address to filter by.

  Returns:
    A list of unique date values.
  """

  try:
    db = client[db_name]
    collection = db[collection_name]

    pipeline = [
      {"$match": {"email": email}},
      {"$unwind": "$chat_history"},
      {"$group": {"_id": "$chat_history.date"}},
      {"$project": {"_id": 0, "date": "$_id"}}
    ]

    unique_dates = []
    for doc in collection.aggregate(pipeline):
      unique_dates.append(doc['date'])

    return unique_dates

  except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.NetworkTimeout, pymongo.errors.ConnectionFailure) as e:
      print(f"Error connecting to MongoDB: {e}")
      print(client)
      return []

def get_guest_chat_history(mongo_client, database_name, collection_name):
    """Retrieves chat history for guest users."""

    db = mongo_client[database_name]
    collection = db[collection_name]

    pipeline = [
        {
            "$match": {
                "email": {"$exists": False}
            }
        }
    ]

    cursor = collection.aggregate(pipeline)
    guest_chat_history = list(cursor)

    return guest_chat_history
