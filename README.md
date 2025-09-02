#  SailPoint Knowledge Bot

This project leverages Retrieval-Augmented Generation (RAG) to provide quick, accurate, and context-aware answers from SailPoint PDF documents.
By combining HuggingFace embeddings, ChromaDB, and ChatGroq LLM (Llama 3.1-8B), the system ensures efficient document parsing, privacy-preserving search, and reliable summarization.

---
 # Documentation
 ## Introduction

The RAG-Powered Document Assistant is designed to overcome the challenge of manually scanning large SailPoint documents.
It enables users to query PDFs in natural language and receive source-backed, precise answers in real time.

---
# Features

- PDF ingestion with automatic text chunking.

- Contextual retrieval using HuggingFace embeddings.

- Vector search with ChromaDB for relevant document sections.

- Answer generation using ChatGroq’s Llama 3.1-8B model.

- Interactive Streamlit chat interface.

- Privacy-first: documents are processed locally, not shared externally.

- Multi-session memory for seamless Q&A continuation.
  
--- 
##  Project Structure

          ┌─────────────┐
          │   User UI   │
          │ (Chat Input)│
          └──────┬──────┘
                 │  Query
                 ▼
        ┌──────────────────┐
        │    ChromaDB      │
        │ (Vector Storage) │
        └──────┬───────────┘
               │ Relevant Docs
               ▼
    ┌─────────────────────────┐
    │ HuggingFace Embeddings  │
    │ (Contextual Retrieval)  │
    └──────────┬─────────────┘
              │ Retrieved Context
              ▼
       ┌───────────────────┐
       │   ChatGroq LLM    │
       │  (Llama-3.1-8B)   │
       └─────────┬─────────┘
                 │  Answer
                 ▼
          ┌─────────────┐
          │   User UI   │
          │ (Final Ans) │
          └─────────────┘

## Models and Techniques

- **Embedding Model** : HuggingFace intfloat/multilingual-e5-large

- **Vector Database** : ChromaDB (persistent storage)

- **LLM** : ChatGroq (Llama-3.1-8B) for final answer generation

- **Frameworks** : LangChain + Streamlit
---

---

## Advantages

- **Faster research** – Navigate SailPoint docs in seconds.  

- **High accuracy** – Context-aware retrieval minimizes hallucination.  

- **Privacy-first** – Data never leaves local storage.  

- **Modular design** – Can extend to other enterprise documents.  

- **Interactive UI** – Easy adoption by technical and non-technical users.  

---
## Disadvantages

- **Storage Intensive** : Vector DB may grow large with many PDFs.

- ** Setup Required** : Needs Python environment and dependencies.

- **Resource Usage** : Heavy models require good hardware for efficiency.

---

## Conclusion  

This RAG-powered document assistant provides a secure, accurate, and efficient way to interact with enterprise knowledge bases.  
By combining **ChromaDB for storage**, **HuggingFace embeddings for contextual retrieval**, and **ChatGroq’s LLM for reasoning**,  
the app ensures faster insights, reduced research time, and privacy-first document handling.  

It is a modular foundation that can be extended to other enterprise use cases such as compliance, legal, or technical support.  

---



