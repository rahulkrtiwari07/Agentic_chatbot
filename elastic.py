import streamlit as st
from elasticsearch import Elasticsearch
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore
from pypdf import PdfReader
import tempfile
from langchain_text_splitters import CharacterTextSplitter
# Load environment variables
load_dotenv('.env.example')

# Set up Elasticsearch connection
es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "qtdrUi2ZMyPcE=m6gO*H"),
        ca_certs="/Users/ripeshghimire/coding/execfiles/elasticsearch-8.14.0/config/certs/http_ca.crt"
    )

# Define the index name
index_name = "vector_index"

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to perform similarity search
def similarity_search(query, chunked_text):
    elastic_vector_search = ElasticsearchStore.from_texts(
        texts=chunked_text,
        index_name=index_name,
        embedding=embedding,
        es_connection=es,
        strategy=DenseVectorStrategy()
    )
    # elastic_vector_search.client.indices.refresh(index=index_name)
    response = elastic_vector_search.similarity_search(query)
    return response[0].page_content

# Streamlit app
st.title("Elastic Vector Search with GPT Conversation Chain")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = []
        for uploaded_file in uploaded_files:
            temp_path = f"{temp_dir}/{uploaded_file.name}"
            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            pdf_paths.append(temp_path)
        
        # Extract and process text from uploaded PDFs
        pdf_text = get_pdf_text(pdf_paths)
        chunked_text = get_text_chunks(pdf_text)
        
        # User input for the query
        query = st.text_input("Enter your query:", "What is the total fertility rate")

        # Display the search results
        if query:
            response = similarity_search(query, chunked_text)
            st.write("Search Results:", response)

        # Conversation chain
        llm = LangChainOpenAI()
        prompt_template = ChatPromptTemplate("Your question: {input}\nAI answer: {output}")
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, prompt_template=prompt_template)

        # User input for conversation
        conversation_input = st.text_input("Ask something to GPT:", "")

        # Display the GPT response
        if conversation_input:
            conversation_response = conversation_chain(conversation_input)
            st.write("GPT Response:", conversation_response)
else:
    st.write("Please upload PDF files to proceed.")
