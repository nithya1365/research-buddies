# Research-Buddies
**Automated Research Paper Finder + RAG QA**

Research-Buddies is a Python tool that serves as a complete academic research assistant, automating literature review, paper retrieval, and RAG-based question answering using modern LLMs.

---

##  Overview

Research-Buddies helps students, researchers, and engineers by streamlining:

-  **Research paper discovery** from multiple sources  
-  **Automatic download** of open-access PDFs  
-  **PDF → text conversion** for processing  
-  **Vector database creation** for semantic search  
-  **RAG-based question answering** using Groq LLMs  
-  **Simple and fast UI** via Gradio  

It reduces manual effort and significantly speeds up the research workflow.

---

##  Features

###  1. Multi-Source Research Paper Search

Fetches papers from multiple APIs:

| Source | Method |
|--------|--------|
| arXiv | REST API |
| Semantic Scholar | Graph API |
| CORE | API (key required) |
| Google Scholar | SerpAPI |

---

### 📈 2. Intelligent Relevance Ranking

Papers are ranked using:

- Query-term frequency  
- Weighted title and abstract matches  
- Fuzzy similarity using `SequenceMatcher`  

---

###  3. Smart PDF Downloader

Automatically avoids restricted or paid sources:

 IEEE Xplore  
 Springer  
 Elsevier  
 MDPI  
 Nature  
 ResearchGate  

Only direct **open-access PDFs** are downloaded.

---

###  4. PDF → Text Processing

Downloaded PDFs are converted to `.txt` for:

- Embedding  
- Chunking  
- RAG context preparation  

---

###  5. Vector Database (ChromaDB)

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
- Persistent Chroma storage  
- Fast cosine similarity-based retrieval  

---

###  6. RAG QA with Groq LLMs

Uses **Mistral-Saba-24B** for accurate and fast responses:

- Answers generated strictly from retrieved documents  
- Prevents hallucinations  
- Uses a hybrid semantic + keyword retrieval pipeline  

---

###  7. Gradio UI

Provides a **user-friendly interface** with:

- Search bar for queries  
- Paper listing and previews  
- RAG-based question answering  

Clean, simple, and beginner-friendly.
