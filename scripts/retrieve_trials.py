import chromadb
import requests
import os

hf_api = os.getenv("hf_inf")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction" 
headers = {"Authorization": f"Bearer {hf_api}"}

def get_query_embedding(query: str):

    response = requests.post(API_URL, headers=headers, json={"inputs": query})

    if response.status_code != 200:
        raise Exception(f"HF API error: {response.text}")

    embedding = response.json()

    # Flatten output (sometimes hf returns nested list)
    if isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding

def load_chroma():
    client = chromadb.PersistentClient(path="vector_db")
    return client.get_or_create_collection("clinical_trials")

collection = load_chroma()

def is_analytical(query: str) -> bool:
    keywords = ["how many", "count", "total", "number of"]
    return any(kw in query.lower() for kw in keywords)

def semantic_search(query: str, top_k=10):
    
    query_vector = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            **meta,
            "chunk_text": doc,
            "relevance_score": round(1 - dist, 4)
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]

def analytical_count(query: str):

    query_vector = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=100,
        include=["metadatas"]
    )

    metas = results["metadatas"][0]
    unique_trials = set(m["trial_id"] for m in metas if m.get("trial_id"))

    return len(unique_trials), list(unique_trials)

def retrieve(query: str):
    if is_analytical(query):
        count, trial_ids = analytical_count(query)
        examples = semantic_search(query, top_k=5)

        return {
            "type": "analytical",
            "count": count,
            "examples": examples
        }

    else:
        results = semantic_search(query, top_k=10)

        return {
            "type": "descriptive",
            "results": results
        }
