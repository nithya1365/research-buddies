import gradio as gr
import requests
import feedparser
import os
import re
import time
from difflib import SequenceMatcher
import subprocess

# Ensure all required packages are installed
def install_required_packages():
    try:
        import pkg_resources
        # Check if feedparser is installed
        pkg_resources.get_distribution('feedparser')
        # Check if serpapi is installed
        pkg_resources.get_distribution('google-search-results')
    except (ImportError, pkg_resources.DistributionNotFound):
        print("Installing required packages...")
        os.system('pip install feedparser google-search-results')

# Install required packages at startup
install_required_packages()

# Now import serpapi after ensuring it's installed
from serpapi import GoogleSearch

# ==== ARXIV SEARCH ====
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

# ==== SEMANTIC SCHOLAR SEARCH ====
def search_semantic_scholar(query, max_results=10):
    papers = []
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,abstract,url,openAccessPdf"
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

# ==== CORE SEARCH ====
def search_core_api(query, api_key, max_results=10):
    papers = []
    try:
        if not api_key:
            print("CORE API key is missing")
            return papers
            
        headers = {'Authorization': f'Bearer {api_key}'}
        url = f'https://core.ac.uk:443/api-v2/search/{query}?page=1&pageSize={max_results}&metadata=true&fulltext=true'
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

# ==== SERP API (GOOGLE SCHOLAR) SEARCH ====
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
            
            # Look for PDF links in resources
            if resources:
                for resource in resources:
                    link = resource.get("link", "")
                    if link and link.lower().endswith(".pdf"):
                        # Skip IEEE papers by checking the URL
                        if "ieeexplore.ieee.org" in link.lower():
                            print(f"Skipping IEEE paper: {result.get('title', '')}")
                            pdf_url = ""  # Skip this paper
                            break
                        pdf_url = link
                        break
            
            # If no PDF found but there's a link, use it anyway
            if not pdf_url and result.get("link"):
                pdf_url = result.get("link")
            
            # Skip IEEE papers
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

# ==== BASIC RANKING FUNCTION (No ML dependencies) ====
def text_similarity(text1, text2):
    """Calculate text similarity using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def rank_papers(query, papers, top_k=10):
    if not papers:
        return []
        
    try:
        print("Ranking papers by relevance...")
        query_terms = query.lower().split()
        
        # Create a list to store paper scores
        paper_scores = []
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            # Skip papers without a title
            if not title:
                continue
            
            # Calculate a simple relevance score
            score = 0
            
            # 1. Count query terms in title and abstract (with higher weight for title)
            for term in query_terms:
                if term in title:
                    score += 3  # Higher weight for title matches
                if term in abstract:
                    score += 1  # Lower weight for abstract matches
            
            # 2. Add similarity score between query and title
            title_sim = text_similarity(query, title)
            score += title_sim * 5  # Weight title similarity highly
            
            # 3. Add similarity score between query and abstract
            if abstract:
                abstract_sim = text_similarity(query, abstract)
                score += abstract_sim * 2
            
            # Store paper with its score
            paper_scores.append((paper, score))
        
        # Sort papers by score in descending order
        paper_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k papers
        ranked_papers = [p for p, _ in paper_scores[:top_k]]
        print(f"Ranked {len(ranked_papers)} papers by relevance")
        return ranked_papers
    
    except Exception as e:
        print(f"Ranking error: {e}")
        return papers[:top_k]  # Fallback to first results

# ==== PDF DOWNLOADER ====
def download_pdfs(papers, folder='downloads', target_successful_downloads=15):
    if not papers:
        print("No papers to download")
        return 0
        
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    successful_downloads = 0
    total_attempts = 0
    
    # List of domains to skip (e.g., paywalls, login-required, etc.)
    skip_domains = [
        "ieeexplore.ieee.org",
        "researchgate.net",
        "link.springer.com",
        "mdpi.com",
        "nature.com",
        "pmc.ncbi.nlm.nih.gov"
    ]
    
    while successful_downloads < target_successful_downloads and total_attempts < len(papers):
        paper = papers[total_attempts % len(papers)]  # Loop through papers
        
        pdf_url = paper['pdf_url']
        
        # Skip known problematic domains
        if any(domain in pdf_url.lower() for domain in skip_domains):
            print(f"Skipping paper from a restricted domain: {paper['title']}")
            total_attempts += 1
            continue
        
        # Ensure the URL ends with '.pdf' for direct PDF links
        if not pdf_url.lower().endswith('.pdf'):
            print(f"Skipping paper with non-PDF link: {paper['title']}")
            total_attempts += 1
            continue
        
        # Clean the title to create a valid filename
        title_clean = re.sub(r'[^\w\s-]', '', paper['title'])
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        
        filename = f"{folder}/{total_attempts+1:02d} - {title_clean[:80]} [{paper['source']}].pdf"
        print(f"Attempting to download from {paper['source']}: {paper['title']}")
        
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200 and response.headers.get('content-type', '').lower() == 'application/pdf':
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Successfully saved to {filename}")
                successful_downloads += 1
            else:
                print(f"✗ Failed to download: {pdf_url} (Status: {response.status_code}, Content-Type: {response.headers.get('content-type')})")
        except Exception as e:
            print(f"✗ Error downloading {pdf_url}: {e}")
        
        total_attempts += 1
        # Add delay to prevent overwhelming APIs
        time.sleep(1)
    
    print(f"\nDownload summary: {successful_downloads} papers successfully downloaded")
    return successful_downloads

# ==== API KEY MANAGEMENT ====
def get_api_keys():
    # Using hardcoded API keys as in the original code
    core_api_key = "qXc7eCZiTfnFuzvsPWoEVjOY1R8G4xa0"  # Default key
    serp_api_key = "25d181450dab71e98838d2c2f6fe93324cb012494e0d77978b3c9eeb8f43bc85"  # Default key
    
    return core_api_key, serp_api_key

# ==== MAIN DRIVER MODIFIED FOR GRADIO ====
def search_and_download_papers(query):
    if not query.strip():
        return "Please enter a search query."
        
    core_api_key, serp_api_key = get_api_keys()
    max_results_per_source = 15
    
    papers = []
    
    # Create a string to capture all output for the Gradio interface
    output_text = "Starting search for research papers...\n\n"
    
    # Search across multiple sources with progress indicators
    output_text += "Searching arXiv...\n"
    papers += search_arxiv(query, max_results=max_results_per_source)
    
    output_text += "Searching Semantic Scholar...\n"
    papers += search_semantic_scholar(query, max_results=max_results_per_source)
    
    output_text += "Searching CORE...\n"
    papers += search_core_api(query, core_api_key, max_results=max_results_per_source)
    
    output_text += "Searching Google Scholar...\n"
    papers += search_google_scholar_serpapi(query, serp_api_key, max_results=max_results_per_source)
    
    # Remove duplicates based on title similarity
    output_text += "Removing duplicate papers...\n"
    seen_titles = set()
    unique_papers = []
    for p in papers:
        title_norm = p['title'].lower().strip()
        if title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_papers.append(p)
    
    output_text += f"Found {len(papers)} papers, {len(unique_papers)} after removing duplicates\n\n"
    
    # Rank and download
    output_text += "Ranking papers by relevance...\n"
    ranked_papers = rank_papers(query, unique_papers, top_k=10)
    
    # Download folder
    download_folder = "downloads"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    output_text += f"Downloading papers to '{download_folder}' folder...\n"
    successful_downloads = download_pdfs(ranked_papers, folder=download_folder, target_successful_downloads=10)

    output_text += f"\nDownload complete: {successful_downloads} papers successfully downloaded to the '{download_folder}' folder."
    
    return output_text

# def launch_second_app():
#     subprocess.Popen(["python", "tempCodeRunnerFile.py"])
#     return "Launching second app..."

# with gr.Blocks() as main_app:
#     gr.Markdown("## Main App")
#     btn = gr.Button("Launch Second App")
#     output = gr.Textbox()
#     btn.click(fn=launch_second_app, outputs=output)


# ==== GRADIO INTERFACE SETUP ====
def create_gradio_interface():
    # Set up the interface with improved UI
    with gr.Blocks(title="Research Paper Downloader") as demo:
        gr.Markdown("# Research Paper Downloader")
        gr.Markdown("Enter keywords or a project title to search for relevant research papers and download them automatically.")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Enter Project Keywords or Title",
                    placeholder="e.g., machine learning for climate change",
                    lines=2
                )
                search_button = gr.Button("Search & Download Papers", variant="primary")
            
        output_box = gr.Textbox(label="Status", lines=15)
        
        search_button.click(fn=search_and_download_papers, inputs=input_text, outputs=output_box)
    
    # Launch the interface
    demo.launch(share=False)

if __name__ == "__main__":
    create_gradio_interface()
    subprocess.run(["python", "chatbot.py"])