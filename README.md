# RAG-based Question Answering System using LangChain, ChromaDB & Ollama

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:
- Loads text from a file  
- Splits it into chunks  
- Generates embeddings  
- Stores vectors in **ChromaDB**  
- Uses **Ollama (Mistral model)** to answer questions based on retrieved context  

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
