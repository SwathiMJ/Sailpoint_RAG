#  SailPoint Knowledge Bot

An intelligent **RAG-powered knowledge assistant** built with **LangChain, LangGraph, Streamlit, and ChromaDB**.  
The bot ingests PDF documents, indexes them with embeddings, and enables **chat-based academic/technical Q&A** with source references.

---

## Problem Statement

Finding accurate information in large technical documents is slow and inefficient with traditional search. The challenge is to build a RAG-powered Knowledge Bot that ingests PDFs, retrieves relevant context, and delivers concise, source-backed answers through an interactive chat interface.

##  Features
- Upload and index PDFs into **ChromaDB** with multilingual embeddings (`intfloat/multilingual-e5-large`).
- Context-aware **question answering** using **Groq’s Llama-3.1-8B-Instant**.
- Multi-session memory with LangGraph.
-  Streamlit-based interactive UI for chatting with documents.
- Privacy-conscious retrieval pipeline.

---

## Tech Stack
- **Frameworks**: [LangChain](https://www.langchain.com/), [LangGraph](https://github.com/langchain-ai/langgraph), [Streamlit](https://streamlit.io/)
- **Vector DB**: [Chroma](https://www.trychroma.com/)
- **Models**: HuggingFace embeddings + Groq Chat Models
- **Storage**: Local persistent ChromaDB

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





