# Insurance AI Assistant - RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot that answers questions from insurance documents.

## Architecture

PDF Documents
↓
Text Chunking
↓
HuggingFace Embeddings
↓
FAISS Vector Database
↓
Retriever
↓
FLAN-T5 LLM
↓
Answer


## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Transformers
- Gradio


## Features

- Query insurance policy documents
- Retrieves relevant information
- Generates grounded answers using RAG