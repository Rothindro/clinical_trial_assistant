from retrieve_trials import retrieve
from llm_generator import generate_answer


def process_user_query(user_query: str) -> dict:

    if not user_query.strip():
        return {
            "answer": "Please enter a valid query.",
            "retrieved_trials": []
        }

    try:
        retrieval_output = retrieve(user_query)
    except Exception as e:
        return {
            "answer": f"Error during retrieval: {str(e)}",
            "retrieved_trials": []
        }

    try:
        answer = generate_answer(user_query, retrieval_output)
    except Exception as e:
        return {
            "answer": f"Error during answer generation: {str(e)}",
            "retrieved_trials": []
        }

    query_type = retrieval_output.get("type")
    display_trials = (
        retrieval_output.get("examples", [])
        if query_type == "analytical"
        else retrieval_output.get("results", [])
    )

    structured_output = []
    for trial in display_trials:
        structured_output.append({
            "trial_id":      trial.get("trial_id", ""),
            "phase":         trial.get("phase", ""),
            "status":        trial.get("status", ""),
            "sponsor":       trial.get("sponsor", ""),
            "interventions": trial.get("interventions", ""),
            "enrollment":    trial.get("enrollment", ""),
            "relevance_score": trial.get("relevance_score", ""),
            "chunk_text":    trial.get("chunk_text", "")[:300]
        })

    print(f"[INFO] Query type: {query_type}")
    print(f"[INFO] Trials for display: {len(structured_output)}")

    return {
        "answer": answer,
        "retrieved_trials": structured_output,
        "query_type": query_type
    }

if __name__ == "__main__":
    print("=== Clinical Trial RAG Assistant ===\n")
    user_query = input("Enter your query: ")
    output = process_user_query(user_query)
    print("\n=== ANSWER ===\n")
    print(output["answer"])