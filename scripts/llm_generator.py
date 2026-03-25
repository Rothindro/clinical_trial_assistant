from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("llama318b"))

def build_context(trials: list, max_trials=3) -> str:
    
    context = ""
    for i, trial in enumerate(trials[:max_trials]):
        # Limit chunk to 150 words for token safety
        chunk = " ".join(trial.get("chunk_text", "").split()[:150])
        context += f"--- Trial {i+1} ---\n{chunk}\n\n"
    return context


def generate_answer(query: str, retrieval_output: dict) -> str:

    query_type = retrieval_output.get("type")
    if query_type == "analytical":
        count = retrieval_output.get("count", 0)
        examples = retrieval_output.get("examples", [])
        context = build_context(examples)

        prompt = f"""You are a clinical research expert assistant.

The user asked an analytical question about NSCLC clinical trials.
Based on semantic search, approximately {count} unique trials matched this query.

Example matching trials:
{context}

User Query: {query}

Instructions:
- Answer directly with the count
- Briefly mention 1-2 example trials if relevant
- Be concise and factual

Answer:"""

    else:
        results = retrieval_output.get("results", [])
        context = build_context(results)
        prompt = f"""You are a clinical research expert assistant.

Answer the user query based strictly on the clinical trial data below.

Instructions:
- Only include trials directly relevant to the query
- For each relevant trial mention: Trial ID, Sponsor, Phase, Status, Drug/Intervention, brief summary
- If no relevant trials found say "No relevant trials found for your query"
- Be precise and factual

Clinical Trial Data:
{context}

User Query: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content