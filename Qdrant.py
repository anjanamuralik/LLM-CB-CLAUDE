import json
import torch
import uuid
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel

# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
model = AutoModel.from_pretrained("BAAI/bge-small-en")

# Function to generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.numpy().flatten()

# Load the JSON data
json_file_path = "C:/Users/AnjanaMurali/Desktop/WEBPAGE_CB/queries.json"  # Replace with your actual file path
with open(json_file_path, "r") as f:
    queries_data = json.load(f)

# Qdrant client setup
qdrant_client = QdrantClient("localhost", port=6333)  # Replace with your Qdrant host and port

# Define collection name and vector size
collection_name = "queries_vectorization"
vector_size = 384  # Based on the embedding model's output size

# Create Qdrant collection
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config={"size": vector_size, "distance": "Cosine"}
)

# Vectorize and upload data
for query_details in queries_data:
    header = query_details["header"]
    description = query_details["description"]
    query = query_details["query"]
    metadata = query_details.get("metadata", {})  # Fetch metadata if available

    # Generate embedding for the description
    embedding_vector = generate_embeddings(description)

    # Prepare payload
    payload = {
        "header": header,
        "description": description,
        "query": query,
        **metadata  # Include metadata fields
    }

    # Insert data into Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": str(uuid.uuid4()),  # Unique ID for each point
                "vector": embedding_vector.tolist(),  # Convert vector to list
                "payload": payload  # Include metadata
            }
        ]
    )

print(f"Data from JSON file has been successfully vectorized and uploaded to Qdrant collection '{collection_name}'.")
