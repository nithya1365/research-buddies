.

🚀 Research-Buddies
Automated Research Paper Finder + RAG QA + Plagiarism Checker

A complete end-to-end academic research assistant that automates literature review, research paper retrieval, vector search, and plagiarism detection using modern LLMs.

📌 Overview

Research-Buddies helps students, researchers, and engineers by automating the entire research workflow:

🔍 Search research papers across multiple databases

📥 Auto-download 10–15 open-access PDFs

🧹 Filter duplicates & non-PDF links

📄 Convert PDFs → text

🧠 Build a vector database using Chroma

🔍 Perform RAG-based question-answering using Groq LLMs

🧪 Detect plagiarism using classical algorithms

🎨 User-friendly UI built with Gradio

✨ Features
🔍 1. Multi-Source Paper Search

Retrieves research papers from:

Source	Method
arXiv	REST API
Semantic Scholar	Graph API
CORE	API (key required)
Google Scholar	SerpAPI
📈 2. Intelligent Relevance Ranking

The system ranks papers using:

Query-term frequency

Weighted matching on Title & Abstract

SequenceMatcher fuzzy similarity

📥 3. Smart PDF Downloader

Automatically avoids blocked/paid sources:

❌ IEEE Xplore
❌ Springer
❌ Elsevier
❌ MDPI
❌ Nature
❌ ResearchGate

Only valid, open-access PDF URLs are downloaded.

📄 4. PDF → Text Processing

All PDFs are converted to .txt for:

Chunking

Embedding

RAG pipeline

🟦 5. Vector Database (ChromaDB)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Persistent Chroma collection

Fast cosine-similarity retrieval

🧠 6. RAG QA with Groq LLMs

Uses Mistral-Saba-24B on Groq for high-speed inference.

Answers strictly from your documents

Prevents hallucinations

Uses contextual chunk retrieval

🧪 7. Plagiarism Checker

Implements classical NLP algorithms:

Shingling (k-grams)

Jaccard Similarity

Rabin-Karp pattern matching

🎨 8. Gradio UI

Simple interface with:

Search bar

Paper results

RAG Question Answering

Plagiarism similarity comparison

🧱 System Architecture
flowchart TD
    A[User Query] --> B[Frontend: Gradio UI]

    B --> C[Search Engine Module]
    C --> C1[arXiv API]
    C --> C2[Semantic Scholar API]
    C --> C3[CORE API]
    C --> C4[SerpAPI - Google Scholar]

    C --> D[Ranking Engine]
    D --> E[Smart PDF Downloader]

    E --> F[PDF → Text Converter]

    F --> G[Text Chunker]
    G --> H[Embedding Generator (MiniLM)]
    H --> I[Chroma Vector DB]

    B --> J[RAG Query]
    J --> K[Retriever from Chroma]
    K --> L[Groq LLM (Mistral-Saba-24B)]
    L --> M[RAG Final Answer]

    B --> N[Plagiarism Checker]
    N --> N1[Shingling]
    N --> N2[Jaccard Similarity]
    N --> N3[Rabin-Karp]

📦 Tech Stack
Frontend

Gradio UI

Backend

Python

LangChain

ChromaDB

HuggingFace Sentence Transformers

Groq LLM API (Mistral-Saba-24B)

APIs

arXiv

Semantic Scholar Graph API

CORE API

SerpAPI (Google Scholar)

📁 Project Structure
research-buddies/
│── search/
│   ├── arxiv_search.py
│   ├── semantic_scholar.py
│   ├── core_search.py
│   └── serpapi_search.py
│
│── downloader/
│   ├── pdf_downloader.py
│   └── duplicate_filter.py
│
│── processing/
│   ├── pdf_to_text.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_db.py
│
│── rag/
│   ├── retriever.py
│   ├── groq_client.py
│   └── rag_pipeline.py
│
│── plagiarism/
│   ├── shingles.py
│   ├── jaccard.py
│   └── rabin_karp.py
│
│── ui/
│   └── app.py
│
└── README.md

▶️ How to Run
git clone https://github.com/yourname/research-buddies
cd research-buddies
pip install -r requirements.txt
python ui/app.py

🔚 Conclusion

This project automates the entire research workflow by integrating:

Multi-source search

Automated PDF retrieval

Text & embedding pipelines

RAG QA using Groq

Plagiarism detection

Perfect for students, researchers, developers, and academic teams.

If you want, I can also:

✅ Generate badges (MIT License, Python version, stars)
✅ Create setup.py or pyproject.toml
✅ Add API keys config template
✅ Write a pitch for interview based on this

Just tell me!

You said:
remove palgirism and project structure in this
ChatGPT said:

Here is the clean, professional README with plagiarism section removed and project structure removed, exactly as you asked.

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

🧱 System Architecture
flowchart TD
    A[User Query] --> B[Frontend: Gradio UI]

    B --> C[Search Engine Module]
    C --> C1[arXiv API]
    C --> C2[Semantic Scholar API]
    C --> C3[CORE API]
    C --> C4[SerpAPI - Google Scholar]

    C --> D[Ranking Engine]
    D --> E[Smart PDF Downloader]

    E --> F[PDF → Text Converter]

    F --> G[Text Chunker]
    G --> H[Embedding Generator (MiniLM)]
    H --> I[Chroma Vector DB]

    B --> J[RAG Query]
    J --> K[Retriever from Chroma]
    K --> L[Groq LLM (Mistral-Saba-24B)]
    L --> M[RAG Final Answer]

📦 Tech Stack
Frontend

Gradio UI

Backend

Python

LangChain

ChromaDB

HuggingFace Sentence Transformers

Groq LLM API (Mistral-Saba-24B)

APIs Used

arXiv

Semantic Scholar Graph API

CORE API

SerpAPI (Google Scholar)

▶️ How to Run
git clone https://github.com/yourname/research-buddies
cd research-buddies
pip install -r requirements.txt
python ui/app.py

🔚 Conclusion

This tool automates:

Research paper fetching

PDF downloading

Text preprocessing

Vector search

RAG-based Q&A

A complete workflow for students, researchers, and engineers looking to speed up literature review and understanding.
