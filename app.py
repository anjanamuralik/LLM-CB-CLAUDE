VECTOR SEARCH STREAMLIT CODE
-----------------------------



import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from anthropic import Anthropic
import yaml
import os

# Load config file
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Initialize models and clients
@st.cache_resource
def initialize_models():
    # Load configuration
    config = load_config()
    
    # BGE model initialization
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
    model = AutoModel.from_pretrained("BAAI/bge-small-en")
    
    # Qdrant client initialization using config
    qdrant_client = QdrantClient(
        host=config['qdrant']['host'],
        port=config['qdrant']['port']
    )
    
    # Claude client initialization using config
    anthropic = Anthropic(
        api_key=config['anthropic']['api_key']
    )
    
    return tokenizer, model, qdrant_client, anthropic, config

def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.numpy().flatten()

def search_query(query_text, tokenizer, model, qdrant_client):
    query_embedding = generate_embeddings(query_text, tokenizer, model)
    
    search_results = qdrant_client.search(
        collection_name="Qdrant_Query_2",  # Replace with your collection name if different
        query_vector=query_embedding.tolist(),
        limit=1
    )
    
    if not search_results:
        return None
    
    if search_results[0].score < 0.8:
        return None
        
    return search_results[0].payload.get("query")

def get_claude_response(anthropic, query, sql_result, config):
    # Include available databases in the context
    db_context = "Available databases: " + ", ".join(config['databases'].keys())
    
    if sql_result:
        prompt = f"""Based on the user's query: "{query}"
        
        {db_context}
        
        I found this SQL query that might help:
        {sql_result}
        
        Please explain how this SQL query addresses their need and what it does."""
    else:
        prompt = f"""The user asked about: "{query}"
        
        {db_context}
        
        I couldn't find a matching SQL query in the database. Please let them know and suggest they rephrase their question or provide more details."""
    
    message = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    return message.content

def main():
    st.title("SQL Query Assistant")
    
    try:
        # Initialize models and clients
        tokenizer, model, qdrant_client, anthropic, config = initialize_models()
        
        # Display available databases
        st.sidebar.header("Available Databases")
        for db in config['databases'].keys():
            st.sidebar.write(f"- {db}")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about an SQL query"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            sql_result = search_query(prompt, tokenizer, model, qdrant_client)
            response = get_claude_response(anthropic, prompt, sql_result, config)
            
            with st.chat_message("assistant"):
                if sql_result:
                    st.markdown(f"```sql\n{sql_result}\n```")
                st.markdown(response)
                
                full_response = f"{sql_result}\n\n{response}" if sql_result else response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
    except yaml.YAMLError as e:
        st.error("Error loading configuration file: " + str(e))
    except Exception as e:
        st.error("An error occurred: " + str(e))

if __name__ == "__main__":
    main()
