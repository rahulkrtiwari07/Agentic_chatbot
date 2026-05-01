PFI Chatbot  Documentation

1.	Purpose of the tool: The tool is build on top of the NFHS data of the government of India and is meant to help in retrieving the data related to state and national information for India on fertility, infant and child mortality, the practice of family planning, maternal and child health, reproductive health, nutrition, anaemia, utilization and quality of health and family planning services.
2.	Dataset and Preprocessing: The dataset on which the chatbot works include:
i)	National Family health survey (State-wise)
ii)	National family health survey (India)
The dataset consists of many charts and graphs which are retrieved from the pdf files and are then pre-processed to make them more understandable to the LLM. The tables in the NFHS pdfs are replaced by NFHS factsheet which are also pre-processed.

3.	Environment:
   
i)	Server specs:
ii)	Python 3.9
iii)	Libraries:
Libraries	Version
Langchain	0.3
Langchain_openai	0.2
Langchain_core	0.3
Langchain_experimental	0.3
Langchain_community	0.3
Langchain-elasticsearch	0.3
Openai	1.55.3
Pypdf	4.2.0
Elasticsearch	8.14.0
Streamlit	1.30.0
Fastapi	0.111.0
Uvicorn	0.30.1
Third party libraries:
Docker Image	Image_name
MongoDB	Mongo:6
Elasticsearch	docker.elastic.co/elasticsearch/elasticsearch:8.14.0

5.	Passwords and details:
OPENAI_API_KEY	'sk-proj-fjchS4MpVTn6WnuQx4x8T3BlbkFJNh1mFXMaJhwP4BPiFN3k'
ELASTICSEARCH_KEY	'elasticsearch123docker'
MONGO_URL	mongodb://root:root12345!!@mongodb:27017/  
GOOGLE_TRANSLATE_API_KEY	“Json_File”

6.	How to deploy: 
i)	Login to the server: 
ssh root@89.116.20.47
Password: Saumyasangal@12
ii)	Install docker on the server -(https://docs.docker.com/desktop/setup/install/linux/ubuntu/)
iii)	Go to the chat_bot directory:
cd chat_bot
iv)	Pull the code from the gitlab repository:
git pull origin new_pipeline
v)	Run the docker container:
Docker compose up -d --build

Ingestion pipeline
 
Retrieval Pipeline:
 
7.	Steps:
Ingestion Pipeline:
	The text from the pdf-files are extracted using the Adobe pdf extract which returns a JSON file.
	The text from the JSON file is extracted and put in a word file.
	The data in the word file is pre-processed manually to deal with the text extracted from the graphs and diagrams.
	The text from the pre-processed word file is extracted.
	The extracted text is split into chunks using the RecursiveCharacterTextSplitter of Langchain. The chunk size is 1000 and the chunk overlap is 300.
	The chunks are converted into embeddings using the OpenAI’s text-embedding-3-large and are stored in the ElasticSearch vector database.

Retrieval Pipeline:
	The user query, chat history and contextualized prompt are send to the LLM to generate a contextualized query which is converted into vectors using the OpenAI’s text-embedding-3-large and send to the ElasticSearch retriever to retrieve the suitable chunks from the vector database.
	The suitable chunks along with the chat history are again send to the LLM to retrieve the response for the user query.
	The response along with the query is saved in the chat_history.pkl file.
	The chat history of the user is send to MongoDB. 
Deployment Process:
	A docker image of the program is made.
	The docker image along with the docker images of the ElasticSearch and MongoDB database are saved in a docker-compose.yml file
	To deploy the code into the server we need to log in to the server, pull the code and run the docker-compose file in the server.
	With the docker container running in the server the ElasticSearch and MongoDB databases are active and now can be accessed through the server URL and their respective ports.
	The ingestion pipeline can be run and the pdfs can be embedded as vectors into the ElasticSearch vector database.
8.	Limitations:
	The ingestion pipeline is not able to compute the tabular and graphical data from the pdfs and it needs to be pre-processed before embedding them into the vector database.

