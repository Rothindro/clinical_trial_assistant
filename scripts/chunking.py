import pandas as pd

df = pd.read_csv("data\\processed_trials.csv")
print(df["combined_text"].isna().sum())

def simple_chunk(text, max_words=250, overlap=30):
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

        if start >= len(words):
            break

    return chunks

chunks = []
metadata = []

for _, row in df.iterrows():
    trial_chunks = simple_chunk(row["combined_text"])

    for chunk in trial_chunks:
        if len(chunk.split()) < 5:
            continue

        chunks.append(chunk)
        metadata.append({
            "trial_id":        row["NCT Number"],
            "condition":       row["Conditions"],
            "phase":           row["Phases"],
            "status":          row["Study Status"],
            "enrollment":      row["Enrollment"],
            "sponsor":         row["Sponsor"],          # ← added
            "interventions":   row["Interventions"],    # ← added
            "start_year":      row["start_year"],
            "completion_year": row["completion_year"]
        })

chunk_df = pd.DataFrame({"chunk_text": chunks})
metadata_df = pd.DataFrame(metadata)

final_df = pd.concat([chunk_df, metadata_df], axis=1)
final_df["start_year"] = final_df["start_year"].astype("Int64")
final_df["completion_year"] = final_df["completion_year"].astype("Int64")

final_df.to_csv("data\\trial_chunks.csv", index=False)
print(f"Total chunks saved: {len(final_df)}")
print("Saved to data\\trial_chunks.csv")