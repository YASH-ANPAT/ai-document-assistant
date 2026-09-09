# AI Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot for asking questions about uploaded PDF documents using semantic search and an open-weight language model.

## Features

- PDF upload and page-aware text extraction
- Overlapping text chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- Top-k relevant context retrieval
- Grounded question answering
- GPT-OSS 20B through Groq inference
- Source and page references
- Streamlit chat interface
- Hallucination-aware prompting

## Architecture

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Sentence Transformer Embeddings
 ↓
FAISS Vector Index
 ↓
Semantic Retrieval
 ↓
Relevant Context
 ↓
GPT-OSS 20B via Groq
 ↓
Grounded Answer