# AI-Based Intelligent Search for ERP Systems

An AI-powered intelligent search system designed to improve information retrieval from ERP datasets using Natural Language Processing and semantic search techniques.

## Project Overview

Traditional keyword-based search systems depend heavily on exact word matching. This project uses semantic search to understand the meaning of a user's query and retrieve the most relevant ERP information.

## Technologies Used

* Python
* Pandas
* NumPy
* Natural Language Processing (NLP)
* Sentence Transformers
* all-MiniLM-L6-v2
* Sentence Embeddings
* Cosine Similarity
* FAISS
* Streamlit
* Scikit-learn

## How It Works

1. ERP data is loaded and preprocessed.
2. Text data is cleaned and prepared for searching.
3. Sentence Transformer `all-MiniLM-L6-v2` converts text into embeddings.
4. User queries are also converted into embeddings.
5. Cosine similarity and FAISS are used to find semantically similar results.
6. The most relevant results are returned to the user.
7. A Streamlit interface provides an easy way to search the ERP data.

## Key Features

* Semantic search instead of only exact keyword matching
* Fast vector similarity search using FAISS
* Natural-language query support
* Sentence embedding generation
* Relevant Top-K result retrieval
* Interactive Streamlit interface

## Model

The project uses the `all-MiniLM-L6-v2` Sentence Transformer model to convert text into numerical vector representations called embeddings.

These embeddings allow the system to compare the meaning of different sentences and queries.

## Applications

The system can be adapted for searching information in:

* ERP systems
* Retail databases
* Inventory systems
* Product databases
* Order management systems
* Customer information systems

## Author

**Salmanul Faris**

GitHub: `salman-209`
