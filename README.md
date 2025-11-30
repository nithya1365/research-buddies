🚀 Research-Buddies
Automated Research Paper Finder + RAG QA

An end-to-end academic research assistant that automates literature review, research paper retrieval, and RAG-based question answering using modern LLMs.

📌 Overview

Research-Buddies helps students, researchers, and engineers by automating:

🔍 Research paper discovery

📥 Automatic open-access PDF download

📄 PDF → text conversion

🧠 Vector database creation

🔍 RAG-based question answering using Groq LLMs

🎨 Simple and fast UI via Gradio

It significantly improves research efficiency and reduces time spent manually searching papers.

✨ Features
🔍 1. Multi-Source Research Paper Search

Fetches papers from:

Source	Method
arXiv	REST API
Semantic Scholar	Graph API
CORE	API (key required)
Google Scholar	SerpAPI
📈 2. Intelligent Relevance Ranking

Papers are ranked using:

Query-term frequency

Weighted title & abstract matching

Fuzzy similarity (SequenceMatcher)

📥 3. Smart PDF Downloader

Automatically avoids blocked or paid sources:

❌ IEEE Xplore
❌ Springer
❌ Elsevier
❌ MDPI
❌ Nature
❌ ResearchGate

Only direct, open-access PDFs are downloaded.

📄 4. PDF → Text Processing

All PDFs are converted to .txt for:

Embedding

Chunking

RAG context creation

🟦 5. Vector Database (ChromaDB)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Persistent Chroma storage

Fast cosine similarity retrieval

🧠 6. RAG QA with Groq LLMs

Uses Mistral-Saba-24B from Groq for fast and accurate answers.

Answers generated strictly from retrieved documents

Prevents hallucinations

Uses hybrid retrieval pipeline (semantic + keyword search)

🎨 7. Gradio UI

Provides:

Search bar

Paper listing

RAG-based question answering

Clean, simple, and accessible to beginners.
