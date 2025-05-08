from elasticsearch import Elasticsearch
import ssl

# Define the path to the CA certificate
ca_cert_path = '/root/test1/http_ca.crt'

# Create an SSL context
context = ssl.create_default_context(cafile=ca_cert_path)

context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

client = Elasticsearch(
    "https://localhost:9200",
    ca_certs= ca_cert_path,
    basic_auth=("elastic", "PEvdyBvHqG1FQVDMYPYd")
)

# Initialize the Elasticsearch client with basic_auth
# es = Elasticsearch("https://elastic:123456789@localhost:9200",
#                    ca_certs=False,
#                    verify_certs=False)

# Test the connection
try:
    response = client.info()
    print(response)
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")



indices = client.indices.get_alias()
for index_name in indices.keys():
    print(index_name)
