from langchain_community.vectorstores.lancedb import LanceDB
import lancedb
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, AsyncIterator, Optional
import time
import logging
import os
import pickle
import warnings
from dotenv import load_dotenv


warnings.filterwarnings("ignore")
load_dotenv('.env.example')
es_pass = os.getenv("ELASTICSEARCH_KEY")
model_ip = os.getenv("model_ip")

STORE_FILE = "chat_history.pkl" 

def load_store():
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "rb") as f:
            return pickle.load(f)
    return {}  # Return empty dict if file doesn't exist

def save_store(store):
    with open(STORE_FILE, "wb") as f:
        pickle.dump(store, f)

class Retrieval:
    def __init__(self, db_path="lancedb_data", table_name="covid_data"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        self.store = load_store()

        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo",
            api_version="2023-06-01-preview",
            azure_endpoint="https://container1.openai.azure.com/",
            api_key= "9k9H4skploPnHIXeBJJvf9ZGI3oLPjTxcmk6m1vjot9CjWv7BWQ1JQQJ99BCAC77bzfXJ3w3AAABACOGY3Y7",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )

        self.retriever = self.retrieve_data()

        # Context-aware question reformulation
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
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

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
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

    def retrieve_data(self):
        return LanceDB(
            connection=self.db,
            embedding=self.embedding,
            table_name=self.table_name,
            text_key="text",
            vector_key="vector",
            id_key="id",
            mode="overwrite",  # or 'append' if already populated
            distance="cosine",
            limit=3
        ).as_retriever()

    async def response_llm(self, query: str, email: Optional[str] = None) -> str:
        start_time = time.time()
        config = {}
        if email:
            config["configurable"] = {"session_id": email}

        result = []
        try:
            async for chunk in self.conversational_rag_chain.astream(
                {"input": query},
                config,
            ):
                if answer_chunk := chunk.get("answer"):
                    result.append(answer_chunk)
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            return f"Error processing response chunk: {e}"

        response = "".join(result)

        save_store(self.store)
        return response
    
async def main():
    query = "What is the status of education among women in Karnataka?"
    retrieval = Retrieval(db_path="/data/lancedb")  # or wherever your DB is mounted
    email = "122461.com"
    async for chunk in retrieval.response_llm(query, email):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
