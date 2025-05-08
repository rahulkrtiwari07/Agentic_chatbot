from fastapi import FastAPI, Request, HTTPException, Depends, Response, Cookie, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import warnings
import logging
from datetime import datetime
import os
from stream import Retrieval
from bson import json_util
from fastapi.responses import JSONResponse
from typing import Optional
from uuid import uuid4
from mongooperations import get_mongo_client, store_user_data, retrieve_user_data, archive_data, get_unique_dates_by_email
import json 
import pymongo
#from translation import detect_language, translate
import asyncio
from Googletranslation import detect_language, translate_to_english, translate_to_hindi
import pytz

# Load environment variables
load_dotenv('.env.example')

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# Initialize FastAPI application
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:4200",     # add this
    "http://127.0.0.1:4200",     # keep if you want both
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
es_pass = os.getenv('ELASTICSEARCH_KEY')
ca_certs = os.getenv('ca_certs')
mongo_uri = os.getenv('MONGO_URL')
database_name = "mydatabase"
collection_name = "user_data"
archived_collection_name = "archived_data"

# Initialize MongoDB client
mongo_client = get_mongo_client(mongo_uri)
print(mongo_client)
retrieval = Retrieval(es_pass=es_pass)
retriever = retrieval.retrieve_data()

# In-memory storage for guest user question counts
guest_question_counts = {}

# Initialization at startup
def get_user_id(guest_id: Optional[str] = Cookie(None)) -> str:
    if guest_id is None:
        guest_id = str(uuid4())
    return guest_id

@app.websocket("/query")
async def websocket_endpoint(websocket: WebSocket, guest_id: str = Depends(get_user_id)):
    await websocket.accept()
    try:
        if not guest_id.startswith("guest_"):
            email = guest_id
        else:
            email = guest_id
            guest_count = guest_question_counts.get(guest_id, 0)
            if guest_count >= 5:
                await websocket.send_text("Error: Guest users can only ask 5 questions. Please provide an email or sign in.")
                await websocket.close(code=1008)
                return
            guest_question_counts[guest_id] = guest_count + 1

        while True:
            try:
                data = await websocket.receive_text()
                payload = data # Assuming the client sends the question as plain text directly
                question = payload

                if not question:
                    await websocket.send_text("Error: Question is required.")
                    continue

                query = question

                # Assuming your retrieval.response_llm_streaming function exists
                # and yields text chunks
                import inspect

                async for chunk in retrieval.response_llm(query, email):
                    await websocket.send_text(chunk)

                # After sending all chunks, you might want to send a completion signal
                await websocket.send_text("[COMPLETED]")

                dt = datetime.now(pytz.timezone('Asia/Kolkata'))
                dt_str = dt.strftime("%d-%m-%Y")
                chat_history = {
                    "question": question,
                    "response": "Streamed response", # Indicate it was a streamed response
                    "date": dt_str
                }
                # Store chat history (adapt as needed)
                # if email and not email.startswith("guest_"):
                #     store_user_data(mongo_client, database_name, collection_name, email, chat_history)
                # else:
                #     db = mongo_client[database_name]
                #     collection = db[collection_name]
                #     collection.insert_one(chat_history)

            except WebSocketDisconnect:
                logging.info(f"Client disconnected: {guest_id}")
                break
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await websocket.send_text(f"Error: {e}")

    finally:
        if guest_id.startswith("guest_") and guest_id in guest_question_counts:
            del guest_question_counts[guest_id]
        await websocket.close()

@app.get('/chat_history/{email}')
async def get_chat_history(email: str, date: Optional[str] = Query(None)):
    try:
        '''if date:
            try:
                # Convert date string to datetime object
                date_obj = datetime.strptime(date, "%d-%m-%Y")
            except ValueError as e:
                logging.error(f"Date format error: {e}")
                return JSONResponse(content={"error": "Invalid date format. Please use DD-MM-YY format."}, status_code=400)
        else:
            date_obj = None'''
        
        history = retrieve_user_data(mongo_client, database_name, collection_name, email, date)
        if not history:
            return JSONResponse(content={"message": "No chat history found for this email."}, status_code=400)
        
        # Use json_util to handle MongoDB-specific types like datetime
        serialized_history = json.loads(json_util.dumps(history))
        return JSONResponse(content={"chat_history": serialized_history})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    
@app.post('/archive')
async def archive_old_data(days_threshold: int = 180):
    try:
        archive_data(mongo_client, database_name, collection_name, archived_collection_name, days_threshold)
        return JSONResponse(content={"message": "Data archived successfully."})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/unique_dates/{email}")
async def get_unique_dates(email: str):
    """Retrieves unique dates for a given email."""

    '''db_name = "your_database_name"
    collection_name = "your_collection_name"'''

    try:
        unique_dates = get_unique_dates_by_email(mongo_client, database_name, collection_name, email)
        return {"unique_dates": unique_dates}
    except (pymongo.errors.PyMongoError, Exception) as e:
        print(f"Error retrieving unique dates: {e}")
        return {"error": "Internal server error"}, 500
    
@app.get("/guest_chat_history")
async def get_guest_chat_history():
    try:
        mongo_client = get_mongo_client(mongo_uri)
        db = mongo_client[database_name]
        collection = db[collection_name]

        pipeline = [
            {
                "$match": {
                    #"email": {"$exists": False}
                    "email": {"$not": {"$regex": "@"}}
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        guest_chat_history = list(cursor)

        # Convert MongoDB objects to JSON for serialization
        serialized_history = json.loads(json_util.dumps(guest_chat_history))

        return {"guest_chat_history": serialized_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving guest chat history: {str(e)}")
