from langchain_elasticsearch import ElasticsearchRetriever
from typing import Dict
from langchain_community.embeddings import OpenAIEmbeddings
from elasticsearch import Elasticsearch
import logging
import os 
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA,VectorDBQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
import warnings 
import tiktoken
warnings.filterwarnings("ignore")
load_dotenv('.env.example')
llm = OpenAI()
es_pass = os.getenv("ELASTICSEARCH_KEY")
model_ip = os.getenv("model_ip")
class Retrieval:
    def __init__(self, es_pass, index_name="pdf_name5"):
        es = Elasticsearch(
            "http://89.116.20.47:9300",
            basic_auth=('elastic', es_pass),
            request_timeout=10,
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = index_name
        self.es = es
        self.embedding = OpenAIEmbeddings()
        if es.ping():
            logging.info('Connection with the database established.')
        else:
            logging.error('Failed to connect to Elasticsearch.')

    def vector_query(self, search_query: str) -> Dict:
        try:
            vector = self.embedding.embed_query(search_query)  
            return {
                "knn": {
                    "field": "vector",
                    "query_vector": vector,
                    "k": 10,
                    "num_candidates": 10
                }
            }
        except KeyError as e:
            print(f"KeyError: {e}")
            return {}

    def retrieve_data(self):
        self.retriever = ElasticsearchRetriever(
            es_client=self.es,
            index_name=self.index_name,
            body_func=self.vector_query,
            content_field='text'
        )
        return self.retriever

    def response_llm(self, query: str) -> str:
        # Initialize the language model
        llm = OpenAI(temperature=0.8)
        
        # Create a custom prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                You are an Advanced AI assistant developed by population foundation of India. You need to provide brief responses to the questions asked.
                If you are not sure about the context of the question ask the user to ellaborate the question.
                If you find the question to be out of context then reply "This is out of scope for this chatbot. Please try to reframe the question with additional keywords and information."
                If there is no reference to a state in the context or there are multiple state in the context then ask the user to specify the state."
                \n
                

                Context: {context}
                \n
                Question: {question}
            """
        )

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
        

        # Create a custom LLMChain with your prompt template
        #llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

        # Create a QA chain
        qa_chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt_template)

        # Create the RetrievalQA object
        qa = RetrievalQA(
            combine_documents_chain=qa_chain,
            retriever=self.retriever
        )

        def count_tokens(chain, query):
            with get_openai_callback() as cb:
                result = chain.run(query)
                print(f'Spent a total of {cb.total_tokens} tokens')

            

        
        # Run the inference
        response = qa({"query": query})

        # Get the updated memory 
        #all_history = memory.load_memory_variables({})
        updated_memory = qa_chain.memory.buffer

        count_tokens(qa, query)

        return response['result'], updated_memory

if __name__ == "__main__":
    query = "राष्ट्रीय परिवार स्वास्थ्य सर्वेक्षण 2019-21 के अनुसार कितने प्रतिशत परिवारों ने पीने के पानी के बेहतर स्रोत का उपयोग किया?"
    retrieval = Retrieval(es_pass=es_pass)
    retriever = retrieval.retrieve_data()
    query1 = "What is this tool for?"
    #response_content = retrieval.response_llm(query1)
    #print(response_content)
    response, updated_memory = retrieval.response_llm(query1)
    print("Response:", response)
    print("Updated Memory:", updated_memory) 
   
    