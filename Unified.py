import os
import re
import time
import requests
import gradio as gr
from difflib import SequenceMatcher
from serpapi import GoogleSearch
import feedparser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

db = None

def get_api_keys():
    return "qXc7eCZiTfnFuzvsPWoEVjOY1R8G4xa0", "25d181450dab71e98838d2c2f6fe93324cb012494e0d77978b3c9eeb8f43bc85"

def search_arxiv(query, max_results=10):
    papers = []
    try:
        base_url = 'http://export.arxiv.org/api/query?'
        search_url = f'{base_url}search_query=all:{query}&start=0&max_results={max_results}'
        response = requests.get(search_url, timeout=10)
        if response.status_code != 200:
            print(f"ArXiv API returned status code {response.status_code}")
            return papers
        feed = feedparser.parse(response.text)
        for entry in feed.entries:
            papers.append({
                'title': entry.title.replace('\n', ' ').strip(),
                'abstract': entry.summary.replace('\n', ' ').strip(),
                'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf',
                'source': 'arXiv'
            })
        print(f"Found {len(papers)} papers from arXiv")
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    return papers

def search_semantic_scholar(query, max_results=10):
    papers = []
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query= {query}&limit={max_results}&fields=title,abstract,url,openAccessPdf"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for paper in data.get("data", []):
                pdf_url = paper.get("openAccessPdf", {}).get("url", "")
                if pdf_url:
                    papers.append({
                        "title": paper["title"].replace('\n', ' ').strip(),
                        "abstract": paper.get("abstract", "").replace('\n', ' ').strip(),
                        "pdf_url": pdf_url,
                        "source": "Semantic Scholar"
                    })
            print(f"Found {len(papers)} papers from Semantic Scholar")
        else:
            print(f"Semantic Scholar API returned status code {res.status_code}")
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
    return papers

def search_core_api(query, api_key, max_results=10):
    papers = []
    try:
        if not api_key:
            print("CORE API key is missing")
            return papers
        headers = {'Authorization': f'Bearer {api_key}'}
        url = f'https://core.ac.uk:443/api-v2/search/ {query}?page=1&pageSize={max_results}&metadata=true&fulltext=true'
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for item in data.get('data', []):
                if item.get("downloadUrl"):
                    papers.append({
                        "title": item.get("title", "").replace('\n', ' ').strip(),
                        "abstract": item.get("description", "").replace('\n', ' ').strip(),
                        "pdf_url": item.get("downloadUrl"),
                        "source": "CORE"
                    })
            print(f"Found {len(papers)} papers from CORE")
        else:
            print(f"CORE API returned status code {res.status_code}")
    except Exception as e:
        print(f"Error searching CORE: {e}")
    return papers

def search_google_scholar_serpapi(query, serp_api_key, max_results=10):
    papers = []
    try:
        if not serp_api_key:
            print("SerpAPI key is missing")
            return papers
        search = GoogleSearch({
            "engine": "google_scholar",
            "q": query,
            "api_key": serp_api_key
        })
        results = search.get_dict()
        for result in results.get("organic_results", [])[:max_results]:
            resources = result.get("resources", [])
            pdf_url = ""
            if resources:
                for resource in resources:
                    link = resource.get("link", "")
                    if link and link.lower().endswith(".pdf"):
                        if "ieeexplore.ieee.org" in link.lower():
                            print(f"Skipping IEEE paper: {result.get('title', '')}")
                            pdf_url = ""
                            break
                        pdf_url = link
                        break
            if not pdf_url and result.get("link"):
                pdf_url = result.get("link")
            if "ieeexplore.ieee.org" in pdf_url.lower():
                print(f"Skipping IEEE paper: {result.get('title', '')}")
                continue
            if pdf_url:
                papers.append({
                    "title": result.get("title", "").replace('\n', ' ').strip(),
                    "abstract": result.get("snippet", "").replace('\n', ' ').strip(),
                    "pdf_url": pdf_url,
                    "source": "Google Scholar"
                })
        print(f"Found {len(papers)} papers from Google Scholar")
    except Exception as e:
        print(f"Error searching Google Scholar: {e}")
    return papers

def text_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def rank_papers(query, papers, top_k=10):
    if not papers:
        return []
    try:
        print("Ranking papers by relevance...")
        query_terms = query.lower().split()
        paper_scores = []
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            if not title:
                continue
            score = 0
            for term in query_terms:
                if term in title:
                    score += 3
                if term in abstract:
                    score += 1
            title_sim = text_similarity(query, title)
            score += title_sim * 5
            if abstract:
                abstract_sim = text_similarity(query, abstract)
                score += abstract_sim * 2
            paper_scores.append((paper, score))
        paper_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_papers = [p for p, _ in paper_scores[:top_k]]
        print(f"Ranked {len(ranked_papers)} papers by relevance")
        return ranked_papers
    except Exception as e:
        print(f"Ranking error: {e}")
        return papers[:top_k]

def download_pdfs(papers, folder='downloads', target_successful_downloads=10):
    if not papers:
        print("No papers to download")
        return 0
    if not os.path.exists(folder):
        os.makedirs(folder)
    successful = 0
    skip_domains = [
        "ieeexplore.ieee.org", "researchgate.net", "link.springer.com",
        "mdpi.com", "nature.com", "pmc.ncbi.nlm.nih.gov"
    ]
    for i, paper in enumerate(papers[:target_successful_downloads]):
        pdf_url = paper.get("pdf_url")
        title = paper.get("title", f"paper{i}")
        filename = os.path.join(folder, f"{i+1}_{title}.pdf")
        if any(domain in pdf_url.lower() for domain in skip_domains):
            print(f"Skipping restricted paper: {title}")
            continue
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
                with open(filename, 'wb') as f:
                    f.write(response.content)
                successful += 1
                print(f"✓ Saved: {filename}")
            else:
                print(f"✗ Invalid or non-PDF content from: {pdf_url}")
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
    return f"✅ Downloaded {successful} valid papers."

def search_and_download_papers(query):
    if not query.strip():
        return "Please enter a search query."
    core_api_key, serp_api_key = get_api_keys()
    papers = []
    papers += search_arxiv(query, max_results=5)
    papers += search_semantic_scholar(query, max_results=5)
    papers += search_core_api(query, core_api_key, max_results=5)
    papers += search_google_scholar_serpapi(query, serp_api_key, max_results=5)
    seen_titles = set()
    unique_papers = []
    for p in papers:
        title_norm = p['title'].lower().strip()
        if title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_papers.append(p)
    ranked_papers = rank_papers(query, unique_papers, top_k=10)
    status = download_pdfs(ranked_papers, folder="downloads")
    
    # Rebuild vectorstore
    current_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(current_dir, "downloads")
    persistent_directory = os.path.join(current_dir, "db1", "chroma_db1")
    
    os.makedirs(persistent_directory, exist_ok=True)
    if os.path.exists(persistent_directory):
        import shutil
        shutil.rmtree(persistent_directory)
    os.makedirs(persistent_directory, exist_ok=True)

    all_docs = []
    for file in os.listdir(downloads_dir):
        if file.endswith(".pdf"):
            path = os.path.join(downloads_dir, file)
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                all_docs.extend(split_docs)
            except Exception as e:
                print(f"⚠️ Skipping corrupted PDF: {file}, Error: {e}")

    if all_docs:
        global db
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(all_docs, embeddings, persist_directory=persistent_directory)
        db.persist()
        status += "\n✅ Vectorstore built successfully."
    else:
        status += "\n❌ No valid PDFs found to build vectorstore."
    return status


current_dir = os.path.dirname(os.path.abspath(__file__))
downloads_dir = os.path.join(current_dir, "downloads")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_question(query):
    global db
    if not db:
        return "❌ Vectorstore not ready. Please download PDFs first."
    results = db.similarity_search(query, k=3)
    context_parts = []
    sources = set()
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        sources.add(source)
        context_parts.append(f"[{source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)
    prompt = f"""
You are a helpful assistant. Use ONLY the information in the following context to answer the question.
If the answer is not found in the context, say "❌ Answer not found in the document."

Context:
{context}

Question: {query}

Answer:
"""
    response = client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )
    answer = response.choices[0].message.content.strip()
    source_list = ", ".join(sorted(sources)) if sources else ""
    return f"{answer}\n\n📄 *Source(s)*: {source_list}"

def chat_with_memory(query, history):
    answer = ask_question(query)
    history.append((query, answer))
    return history, history, ""

def refresh_pdfs():
    return [
        gr.File(value=os.path.join(downloads_dir, f), label=f)
        for f in os.listdir(downloads_dir)
        if f.lower().endswith(".pdf")
    ]


with gr.Blocks(title="📚 Unified Research Assistant") as demo:
    gr.Markdown("# 📚 Unified Research Assistant")

    with gr.Tabs():
        with gr.Tab("🔍 Search & Download Papers"):
            input_text = gr.Textbox(label="Search Query", placeholder="e.g., machine learning climate change")
            output_box = gr.Textbox(label="Status", lines=8)
            search_button = gr.Button("Search & Download", variant="primary")
            search_button.click(fn=search_and_download_papers, inputs=input_text, outputs=output_box)

        with gr.Tab("📘 Chat with PDFs"):
            gr.Markdown("Ask questions about the downloaded documents.")
            chatbot = gr.Chatbot(label="Chat History")
            query_input = gr.Textbox(lines=2, placeholder="Type your question here...")
            ask_button = gr.Button("Ask")
            history_state = gr.State([])
            ask_button.click(fn=chat_with_memory, inputs=[query_input, history_state], outputs=[chatbot, history_state, query_input])
            gr.Markdown("### 📂 Available PDFs:")
            refresh_button = gr.Button("🔄 Refresh PDF List")
            pdf_output = gr.Column()
            refresh_button.click(fn=refresh_pdfs, outputs=pdf_output)
            demo.load(fn=refresh_pdfs, outputs=pdf_output)


if __name__ == "__main__":
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("db1", exist_ok=True)
    demo.launch(share=False)