import time
from langchain_elasticsearch import ElasticsearchRetriever
from typing import Dict, AsyncIterator, Optional
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from elasticsearch import Elasticsearch
import logging
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import warnings
import tiktoken
import pickle

from langchain_openai import AzureChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI


warnings.filterwarnings("ignore")
load_dotenv('.env.example')
es_pass = os.getenv("ELASTICSEARCH_KEY")
model_ip = os.getenv("model_ip")

STORE_FILE = "chat_history.pkl"  # File to store chat history

def load_store():
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "rb") as f:
            return pickle.load(f)
    return {}  # Return empty dict if file doesn't exist

def save_store(store):
    with open(STORE_FILE, "wb") as f:
        pickle.dump(store, f)


class Retrieval:

    def __init__(self, es_pass, index_name="covid"):
        es = Elasticsearch(
            "http://164.52.193.73:9201",
            basic_auth=('elastic', es_pass),
            request_timeout=60,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = index_name
        self.es = es
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        if es.ping():
            logging.info('Connection with the database established.')
        else:
            logging.error('Failed to connect to Elasticsearch.')
        self.store = load_store()
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo",
            api_version="2023-06-01-preview",
            azure_endpoint="https://container1.openai.azure.com/",
            api_key="9k9H4skploPnHIXeBJJvf9ZGI3oLPjTxcmk6m1vjot9CjWv7BWQ1JQQJ99BCAC77bzfXJ3w3AAABACOGY3Y7",  
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,  # Enable streaming
        )
        self.retriever = self.retrieve_data()
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.system_prompt = (
            "You are an expert for health statistics in India and you work for Microware"
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Also do not perform any calculations"
            "Provide the answers in small and multiple sentences." 
            
            "\n\n"
            "{context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def vector_query(self, search_query: str) -> Dict:
        try:
            vector = self.embedding.embed_query(search_query)
            return {
                "knn": {
                    "field": "vector",
                    "query_vector": vector,
                    "k": 5,
                    "num_candidates": 10
                }
            }
        except KeyError as e:
            print(f"KeyError: {e}")
            return {}

    def retrieve_data(self):
        retriever = ElasticsearchRetriever(
            es_client=self.es,
            index_name=self.index_name,
            body_func=self.vector_query,
            content_field='text'
        )
        return retriever
    
    async def response_llm(self, query: str, email: Optional[str] = None) -> str:
        start_time = time.time()
        # callback_handler = StreamingStdOutCallbackHandler() # Optional: Print to console

        config = {}
        if email:
            config["configurable"] = {"session_id": email}

        result = []  # To accumulate the response chunks

        try:
            async for chunk in self.conversational_rag_chain.astream(
                {"input": query},
                config,
                # callbacks=[callback_handler]
            ):
                if answer_chunk := chunk.get("answer"):
                    result.append(answer_chunk)
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            return f"Error processing response chunk: {e}"

        # Join the response chunks and return the result as a single string
        response = "".join(result)

        save_store(self.store)
        total_time = time.time() - start_time
        #print(f"Total `response_llm` execution time: {total_time:.4f} seconds")

        return response

    def save_store(self):
        with open(STORE_FILE, "wb") as f:
            pickle.dump(self.store, f)

async def main():
    query = "राष्ट्रीय परिवार स्वास्थ्य सर्वेक्षण 2019-21 के अनुसार कितने प्रतिशत परिवारों ने पीने के पानी के बेहतर स्रोत का उपयोग किया?"
    retrieval = Retrieval(es_pass=es_pass)
    query1 = "What is covid?"
    email = "122461.com"
    async for chunk in retrieval.response_llm(query1, email):
        print(f"{chunk}", end="", flush=True) 

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())