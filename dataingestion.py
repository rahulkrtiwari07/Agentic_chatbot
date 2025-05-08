from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import logging
import semchunk
from langchain_community.embeddings import OpenAIEmbeddings
import pdfplumber


load_dotenv('.env.example')
es_pass = os.getenv('ELASTICSEARCH_KEY')

class DataIngestion:
    def __init__(self, es_pass, index_name="mw_chatbot2"):
        es = Elasticsearch (
            "http://89.116.20.47:9300",
            basic_auth=('elastic', es_pass),
            #request_timeout=10,
            verify_certs=False,
        )
        self.index_name = index_name
        self.es = es
        self.store = None
        if es.ping():
            logging.info('Connection with the database established.')
        else:
            logging.error('Failed to connect to Elasticsearch.')

    def get_multiple_pdfs_text(self, pdf_directory):
        '''
        Parameters->
        pdf_directory : directory containing PDF files (str)
        Function-> Takes a directory path as input, reads all PDF files in the directory,
        and returns the concatenated text from all the documents.
        '''
        all_text = ''
        try:
            # List all PDF files in the directory
            for pdf_file in pdf_directory:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                try:
                    text = ''
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                    all_text += text + '\n'
                except FileNotFoundError:
                    logging.error(f"File {pdf_path} not found.")
                except Exception as e:
                    logging.error(f"An error occurred while processing {pdf_path}: {e}")
        except Exception as e:
            logging.error(f"An error occurred while listing files in directory {pdf_directory}: {e}")

        return all_text
    def chunk_text(self, text):
        '''
        Parameters -> 
        text: Takes the text from the pdf
        Function-> Chunks the text based 
        '''
        chunker = semchunk.chunkerify('gpt-4', chunk_size=1000, max_token_chars=1000)
        chunked_text = chunker(text)
        return chunked_text

    def embed_text(self,chunktext):
        '''
        Parameters ->
        chunktext-> Takes a list of chunk text 
        Function-> embeds the list of chunk text by OpenAI Embedding and store it in elastic search store
        '''
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        store = ElasticsearchStore(
            es_connection=self.es,
            embedding=embedding,
            index_name=self.index_name,
            strategy= DenseVectorStrategy(),
            
        )
        store.add_texts(chunktext)
        return store

    def get_data(self, folderpath):
        '''Parameters:
        folderPath: Takes folderpath as parameter
        Function -> takes folderpath and retrieves all the pdf files 
        '''
        files = os.listdir(folderpath)
        pdf_files = [file for file in files if file.lower().endswith('.pdf')]
        return [os.path.join(folderpath, pdf_file) for pdf_file in pdf_files]

    def ingest_data(self, folderpath):
        '''
        Parameters: folderpath: takes the path given 
        Function -> returns the embed function and stores the retrieval 
        '''
        all_text = self.get_multiple_pdfs_text(folderpath)
        chunked_text = self.chunk_text(all_text)
        embed = self.embed_text(chunked_text)
        logging.info("Data Ingestion process completed successfully")
        return embed

if __name__ == '__main__':
    ingestion = DataIngestion(es_pass=es_pass)
    store = ingestion.ingest_data(folderpath='Pdffiles')