from elasticsearch import Elasticsearch

# Replace with your Elasticsearch connection details
es = Elasticsearch([{'host': '89.116.20.47:9300', 'port': 9300}])

# Index name to delete
index_name = 'mw_chatbot2'

try:
    # Delete the index
    es.indices.delete(index=index_name, ignore=[400, 404])
    print(f"Index '{index_name}' deleted successfully.")

except Exception as e:
    print(f"Error deleting index '{index_name}': {e}")