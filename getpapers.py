import requests
import feedparser
import os
import re
import time
import random
from difflib import SequenceMatcher

try:
    from symspellpy import SymSpell, Verbosity
    import pkg_resources
except ImportError:
    print("Installing required packages...")
    os.system('pip install symspellpy')
    from symspellpy import SymSpell, Verbosity
    import pkg_resources

try:
    from serpapi.google_search_results import GoogleSearch

except ImportError:
    print("Installing SerpAPI...")
    os.system('pip install google-search-results')
    from serpapi import GoogleSearch

# ==== SPELL CHECKER FUNCTIONS ====
def init_spell_checker():
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        if not os.path.exists(dictionary_path):
            print("Dictionary not found. Make sure symspellpy is properly installed.")
            return None
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        return sym_spell
    except Exception as e:
        print(f"Error initializing spell checker: {e}")
        return None

def correct_query(sym_spell, query):
    if sym_spell is None:
        return query
    try:
        suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
        return suggestions[0].term if suggestions else query
    except Exception as e:
        print(f"Error correcting query: {e}")
        return query

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
                        pdf_url = link
                        break
            
            # If no PDF found but there's a link, use it anyway
            if not pdf_url and result.get("link"):
                pdf_url = result.get("link")
                
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
def download_pdfs(papers, folder='downloads'):
    if not papers:
        print("No papers to download")
        return
        
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    successful_downloads = 0
    for i, paper in enumerate(papers, start=1):
        pdf_url = paper['pdf_url']
        
        # Clean the title to create a valid filename
        title_clean = re.sub(r'[^\w\s-]', '', paper['title'])
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        
        filename = f"{folder}/{i:02d} - {title_clean[:80]} [{paper['source']}].pdf"
        print(f"Downloading from {paper['source']}: {paper['title']}")
        
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200 and response.headers.get('content-type', '').lower() == 'application/pdf':
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Saved to {filename}")
                successful_downloads += 1
                # Add delay to prevent overwhelming APIs
                time.sleep(1)
            else:
                print(f"✗ Failed to download: {pdf_url} (Status: {response.status_code}, Content-Type: {response.headers.get('content-type')})")
        except Exception as e:
            print(f"✗ Error downloading {pdf_url}: {e}")
    
    print(f"\nDownload summary: {successful_downloads} of {len(papers)} papers successfully downloaded")

# ==== API KEY MANAGEMENT ====
def get_api_keys():
    # Get API keys from environment variables first
    # core_api_key = os.environ.get("CORE_API_KEY")
    # serp_api_key = os.environ.get("SERP_API_KEY")
    
    # If not found in environment, ask user
    
    core_api_key = "qXc7eCZiTfnFuzvsPWoEVjOY1R8G4xa0"  # Default key
    
    
        
    serp_api_key = "25d181450dab71e98838d2c2f6fe93324cb012494e0d77978b3c9eeb8f43bc85"  # Default key
    
    return core_api_key, serp_api_key

# ==== MAIN DRIVER ====
def main():
    print("=== Research Paper Downloader ===")
    
    sym_spell = init_spell_checker()
    raw_topic = input("Enter your project goal or topic: ")
    topic = correct_query(sym_spell, raw_topic)
    print(f"Searching for corrected query: {topic}")
    
    core_api_key, serp_api_key = get_api_keys()
    
    # Ask for max results
    try:
        max_results_per_source = int(input("Enter maximum results per source (default: 5): ") or "5")
    except ValueError:
        max_results_per_source = 5
        print("Invalid input, using default: 5")
    
    papers = []
    
    # Search across multiple sources with progress indicators
    print("\nSearching arXiv...")
    papers += search_arxiv(topic, max_results=max_results_per_source)
    
    print("\nSearching Semantic Scholar...")
    papers += search_semantic_scholar(topic, max_results=max_results_per_source)
    
    print("\nSearching CORE...")
    papers += search_core_api(topic, core_api_key, max_results=max_results_per_source)
    
    print("\nSearching Google Scholar...")
    papers += search_google_scholar_serpapi(topic, serp_api_key, max_results=max_results_per_source)
    
    # Remove duplicates based on title similarity
    print("\nRemoving duplicate papers...")
    seen_titles = set()
    unique_papers = []
    for p in papers:
        # Normalize title for comparison
        title_norm = p['title'].lower().strip()
        if title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_papers.append(p)
    
    print(f"Found {len(papers)} papers, {len(unique_papers)} after removing duplicates")
    
    # Ask how many papers to download
    try:
        top_k = int(input(f"\nHow many top papers to download? (default: 10, max: {len(unique_papers)}): ") or "10")
        top_k = min(top_k, len(unique_papers))
    except ValueError:
        top_k = min(10, len(unique_papers))
        print(f"Invalid input, using default: {top_k}")
    
    # Rank and download
    print("\nRanking papers by relevance...")
    ranked_papers = rank_papers(topic, unique_papers, top_k=top_k)
    
    # Ask for download folder
    download_folder = "downloads"
    
    print(f"\nDownloading {len(ranked_papers)} papers to '{download_folder}'...")
    download_pdfs(ranked_papers, folder=download_folder)
    print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")