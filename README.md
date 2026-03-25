# Clinical Trial Intelligence Assistant (RAG-based AI System)
---

## 📖 Overview
---
The Clinical Trial Intelligence Assistant is an AI-powered system designed to retrieve and analyze clinical trial (NSCLC related trials) data using a hybrid approach combining semantic search and structured filtering.

It leverages:
- Vector databases (ChromaDB)
- Sentence embeddings (SentenceTransformers (all-MiniLM-L6-v2))
- Query analysis for intent detection
- Retrieval-Augmented Generation (RAG) with LLMs

The system enables users to:
- Search clinical trials using natural language
- Perform analytical queries (e.g., phase, status, enrollment)
- Get structured and contextual insights

We use a hybrid Retrieval-Augmented Generation (RAG) pipeline:

1. Query Analysis → Detects intent (analytical vs semantic)
2. Hybrid Retrieval:
   - Semantic Search (embeddings)
   - Metadata Filtering
3. Context Construction
4. LLM-based Answer Generation


## ⚙️ Tech Stack
---

| Component       | Technology                              |
|-----------------|-----------------------------------------|
| Language        | Python                                  |
| Vector DB       | ChromaDB                                |
| Embeddings      | SentenceTransformers (PubMedBERT)       |
| LLM             | Llama 3.1 8B Instant (Groq)             |
| Data Processing | Pandas                                  |
| UI              | Streamlit                               |

## 🔍 Key Features
---

- Semantic search using embeddings
- Hybrid retrieval (vector + metadata)
- Query intent classification
- Structured clinical trial insights
- Scalable RAG pipeline
