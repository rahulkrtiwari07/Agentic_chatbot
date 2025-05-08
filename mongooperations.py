import datetime
import pymongo
from datetime import datetime, timedelta
import pymongo.errors
import bson

def get_mongo_client(uri):
    """Get a MongoDB client."""
    return pymongo.MongoClient(uri)

def store_user_data(client, db_name, collection_name, email, chat_history):
    """Stores user data (email address, datetime, and chat history) in MongoDB."""
    try:
        db = client[db_name]
        collection = db[collection_name]

        chat_document = collection.find_one({"email": email})

        if chat_document:
            collection.update_one(
                {"_id": chat_document["_id"]}, {"$push": {"chat_history": chat_history}}
            )
        else:
            new_document = {"email": email, "chat_history": [chat_history]}
            collection.insert_one(new_document)

        print(f"Message stored for chat: {email}")
        print(chat_document)
    except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.NetworkTimeout, pymongo.errors.ConnectionFailure) as e:
        print(f"Error connecting to MongoDB: {e}")

def retrieve_user_data(client, db_name, collection_name, email, date=None):
    """Retrieves user data based on email address and date."""
    try:
        db = client[db_name]
        collection = db[collection_name]

        if email == "admin_login":
            # If email is "admin_login", return all documents
            pipeline = [
                {"$project": {"_id": 0, "email": 1, "chat_history": 1}}  # Project only email and chat_history
            ]
            chat_history = []
            for doc in collection.find():
                chat_history.append(doc)
            return chat_history

        else:
            # Otherwise, filter by email and date as before
            if date == None:
                pipeline = [
                    {"$match": {"email": email}},  # Filter documents by email ID
                    {"$unwind": "$chat_history"},  # Unwind the chat_history array
                    {"$project": {"_id": 0, "date": "$chat_history.date", "Question": "$chat_history.question", "Response": "$chat_history.response"}}
                ]
                chat_history = []
                for doc in collection.aggregate(pipeline):
                    chat_history.append(doc)
                return chat_history

            else:
                pipeline = [
                    {"$match": {"email": email}},  # Filter documents by email ID
                    {"$unwind": "$chat_history"},  # Unwind the chat_history array
                    {"$project": {"_id": 0, "date": "$chat_history.date", "Question": "$chat_history.question", "Response": "$chat_history.response"}}
                ]

                # Filter by date
                datetime_obj = datetime.strptime(date, "%d-%m-%Y")
                pipeline.append({"$match": {"date": date}})

                chat_history = []
                for doc in collection.aggregate(pipeline):
                    chat_history.append(doc)

                return chat_history

    except pymongo.errors.ConfigurationError as e:
        print(f"Error connecting to MongoDB: {e}")

    except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.NetworkTimeout, pymongo.errors.ConnectionFailure) as e:
        print(f"Error connecting to MongoDB: {e}")

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
