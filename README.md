This repository provides a implementation of Retrieval Augmented Generation (RAG) using the LLaMA-3.1 8b language model, Qdrant vector database, and Langchain framework for question answering on PDFs.

Key Components:

LLaMA-3.1 8b LLM model: A large language model for text generation, accessed via the Groq API.

Qdrant vectordb: A vector database for storing vectors and querying PDFs

Langchain framework: A Python library for building and integrating language models and vector databases

How it Works:

The Qdrant vector database stores the PDFs as dense vectors.

When user asks a question is asked, the RAG model retrieves relevant PDFs from the Qdrant vector database.

The RAG model uses the retrieved PDFs and user query to generate an answer, by calling the LLaMA-3.1 8b language model via the Groq API.
