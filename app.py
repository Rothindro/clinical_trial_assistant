import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import streamlit as st
import pandas as pd
from scripts.driver import process_user_query

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trial RAG Assistant",
    page_icon="🧬",
    layout="centered"
)

# ── Cache model + ChromaDB loading ───────────────────────────────────────────
@st.cache_resource
def initialize_system():
    try:
        from scripts.retrieve_trials import model, collection
        print("✅ Model and collection loaded")
        return model, collection
    except Exception as e:
        print("❌ Error loading system:", e)
        raise e

initialize_system()

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "How many trials are in Phase 2 with pembrolizumab?",
    "Tell me about a BMS sponsored Phase 3 trial",
    "How many completed trials involve immunotherapy?",
    "Give me details about a recruiting trial for NSCLC",
    "How many trials have enrollment greater than 500?",
]

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_trials" not in st.session_state:
    st.session_state.last_trials = []

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 Clinical Trial RAG Assistant")
st.caption("Powered by all-MiniLM-L6-v2 + Llama 3.1 8B via Groq | NSCLC Trials Database")
st.divider()

# ── Example queries on homepage ───────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUERIES):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle example query click ────────────────────────────────────────────────
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")
else:
    user_input = None

# ── Chat input ────────────────────────────────────────────────────────────────
typed_input = st.chat_input("Ask about NSCLC clinical trials...")
if typed_input:
    user_input = typed_input

# ── Process query ─────────────────────────────────────────────────────────────
if user_input:

    # Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching clinical trials database..."):
            output = process_user_query(user_input)

        answer = output.get("answer", "No answer generated.")
        trials = output.get("retrieved_trials", [])

        # Display answer
        st.markdown(answer)

        # Store trials for CSV download
        if trials:
            st.session_state.last_trials = trials

        # Show retrieved trials in expander
        if trials:
            with st.expander(f"📋 View {len(trials)} Retrieved Trial(s)", expanded=False):
                for i, trial in enumerate(trials):
                    st.markdown(f"**Trial {i+1}**")
                    fields = {
                        "Trial ID":     trial.get("trial_id"),
                        "Sponsor":      trial.get("sponsor"),
                        "Phase":        trial.get("phase"),
                        "Status":       trial.get("status"),
                        "Intervention": trial.get("interventions"),
                        "Enrollment":   trial.get("enrollment"),
                        "Relevance":    trial.get("relevance_score"),
                        "Summary":      trial.get("chunk_text"),
                    }
                    for label, value in fields.items():
                        if value and str(value).strip() not in ["", "nan", "None", "-1"]:
                            st.markdown(f"- **{label}:** {value}")
                    st.divider()

    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 About")
    st.info(
        "This assistant answers questions about "
        "Non-Small Cell Lung Cancer (NSCLC) clinical trials "
        "using Retrieval Augmented Generation (RAG)."
    )

    st.markdown("### 📥 Download Results")
    if st.session_state.last_trials:
        df_download = pd.DataFrame(st.session_state.last_trials)
        if "chunk_text" in df_download.columns:
            df_download = df_download.drop(columns=["chunk_text"])
        csv = df_download.to_csv(index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name="retrieved_trials.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("Ask a question first to enable download.")

    st.markdown("### 🗑️ Clear Chat")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_trials = []
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit · ChromaDB · Groq · all-MiniLM-L6-v2")