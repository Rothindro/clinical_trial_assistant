import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

df = pd.read_csv("data\\trial_chunks.csv")
print(f"Total chunks to embed: {len(df)}")

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
client = chromadb.PersistentClient(path="vector_db")

# Safety: delete old collection to avoid duplicate ID issues on re-runs
try:
    client.delete_collection(name="clinical_trials")
    print("Old collection deleted.")
except:
    pass

collection = client.get_or_create_collection(name="clinical_trials")

def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip()

def safe_int(val):
    try:
        if pd.isna(val):
            return -1
        return int(val)
    except:
        return -1

BATCH_SIZE = 64

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[i:i + BATCH_SIZE]
    texts = batch["chunk_text"].tolist()

    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    ids = [str(idx) for idx in batch.index]

    metadatas = [
        {
            "trial_id":        safe_str(row["trial_id"]),
            "condition":       safe_str(row["condition"]),
            "phase":           safe_str(row["phase"]),
            "status":          safe_str(row["status"]),
            "sponsor":         safe_str(row["sponsor"]),
            "interventions":   safe_str(row["interventions"]),
            "enrollment":      safe_int(row["enrollment"]),
            "start_year":      safe_int(row["start_year"]),
            "completion_year": safe_int(row["completion_year"])
        }
        for _, row in batch.iterrows()
    ]

    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)

print(f"✅ Vector DB built successfully! Total chunks indexed: {len(df)}")
print("Model: all-MiniLM-L6-v2")