from typesense import Client
from typesense.exceptions import ObjectNotFound
from dotenv import load_dotenv
import os

load_dotenv()

"""
COLLECTIONS:
songs   -   used for initial tests with CLIP embeddings
songs2  -   used for tests with PCA embeddings  
"""
COLLECTION_NAME="songs2"
COLLECTION_DIMENSIONS=2000

client = Client({
  'nodes': [{
        'host': os.environ.get("TYPESENSE_NODE"),
        'port': os.environ.get("TYPESENSE_PORT"),
        'protocol': 'https'
    }],
  'api_key': os.environ.get("TYPESENSE_API_KEY"),
  'connection_timeout_seconds': 5
})

collection_schema = {
  "name": "songs2",
  "fields": [
    {"name": "song_id", "type": "string"}, # Firestore doc id
    {"name": "file_path", "type": "string"},
    {"name": "embedding", "type": "float[]", "num_dim": COLLECTION_DIMENSIONS} 
  ],
}

# Check if the collection already exists
try:
    # This will throw an ObjectNotFound exception if the collection does not exist
    existing_collection = client.collections[COLLECTION_NAME].retrieve()
    print(f"Collection {COLLECTION_NAME} already exists.")
except ObjectNotFound:
    # If the collection does not exist, create it
    client.collections.create(collection_schema)
    print(f"Created new collection {COLLECTION_NAME}.")

def fetch_embeddings(song_id):
    try:
        print(f"Fetching embeddings for song with id: {song_id}")
        search_parameters = {
            'collection': COLLECTION_NAME,
            'query_by': 'song_id',
            'q': song_id,
            'per_page':1,
            'page':1,
        }
        search_results = client.collections[COLLECTION_NAME].documents.search(search_parameters)

        # Verify that the song was found
        if search_results['found'] == 0 or len(search_results['hits']) == 0:
            print("No docs found")
            raise Exception(f"No docs found")

        if search_results['hits'][0]['document']['song_id'] != song_id:
            print(f"Song with id {song_id} not found.")
            raise Exception(f"Song with id {song_id} not found.")

        return search_results['hits'][0]['document']['embedding']
    except Exception as e:
        print(f"Error: {e}")
        return None

def search_song_by_embedding(embeddings):
    try:
        vector_query = f"embedding:({embeddings}, k:100)"
        # Typesense vector search 
        search_parameters = {
            'searches':[{
                'q':'*',  
                'vector_query': vector_query,
                'per_page':10,
                'page':1,
                'exclude_fields': 'embedding'
            }]
        }
        search_results = client.multi_search.perform(search_parameters, {
            'collection': COLLECTION_NAME
        })

        # Verify that the song was found
        if search_results['results'][0]['found'] == 0:
            raise Exception(f"No docs found")

        return search_results['results'][0]['hits']
        

    except Exception as e:
        print(f"Error: {e}")
        return None


def save_embeddings_to_typesense(song_id, embeddings, file_path):
    print("Saving embeddings to Typesense")
    document = {
        'song_id': song_id,
        'embedding': embeddings,
        'file_path': file_path
    }
    client.collections[COLLECTION_NAME].documents.create(document)
