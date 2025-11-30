# research-buddies
# research-buddies
Research Paper Finder + RAG QA 

Automated academic research assistant using multi-source paper search, downloader, vector search, and plagiarism detection.

🚀 Overview

This project is an end-to-end research automation tool.
It helps students, researchers, and engineers by:

🔍 Searching research papers from arXiv, Semantic Scholar, CORE API, Google Scholar (SerpAPI)

📥 Downloading 10–15 open-access PDFs automatically

🧹 Filtering duplicates, blocked domains, and non-PDF sources

📄 Converting PDFs → Text

🧠 Creating a Chroma vector database

🧩 Running RAG-based QA using Groq LLMs

🧪 Performing plagiarism checking using

Shingling

Jaccard Similarity

Rabin-Karp pattern matching

🎨 User-friendly Gradio UI

This tool automates literature review, summarization, and similarity analysis.

📦 Features
🔍 1. Multi-source Research Paper Search

The system fetches papers from:

Source	Method
arXiv	REST API
Semantic Scholar	Graph API
CORE	API (API Key required)
Google Scholar	SerpAPI
📈 2. Relevance Ranking

Papers are ranked using:

Query-term frequency

Weighted title & abstract matches

SequenceMatcher (fuzzy similarity)

📥 3. Smart PDF Downloader

Avoids:

IEEE Xplore

Springer

Elsevier

ResearchGate

MDPI

Nature

Only direct, open-access, valid PDFs are downloaded.

📂 4. PDF → Text Conversion

All downloaded PDFs are converted into .txt for:

RAG context extraction


🟦 5. Vector Database with ChromaDB

Uses:

sentence-transformers/all-MiniLM-L6-v2 embeddings

Persistent Chroma storage

🧠 6. Groq LLM RAG QA

Uses Mistral-Saba-24B from Groq to generate answers strictly from the documents.

🌟 Tech Stack
Frontend

Gradio UI

Backend

Python

LangChain

ChromaDB

HuggingFace Embeddings

Groq LLMs (Mistral-Saba-24B)

APIs

ArXiv

Semantic Scholar

CORE

SerpAPI (Google Scholar)
