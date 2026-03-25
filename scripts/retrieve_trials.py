import chromadb
import torch
from sentence_transformers import SentenceTransformer

# Auto detect GPU or fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

def load_chroma():
    client = chromadb.PersistentClient(path="vector_db")
    return client.get_or_create_collection("clinical_trials")

model = load_model()
collection = load_chroma()

def is_analytical(query: str) -> bool:
    keywords = ["how many", "count", "total", "number of"]
    return any(kw in query.lower() for kw in keywords)

def semantic_search(query: str, top_k=10):
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
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
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    results = collection.query(query_embeddings=[query_vector.tolist()], n_results=100,include=["metadatas"])
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