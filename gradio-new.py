import os
import re
import gradio as gr
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Load .env for API keys
load_dotenv()

# Define base paths
current_dir = os.path.dirname(os.path.abspath(__file__))
downloads_dir = os.path.join(current_dir, "downloads")
downloads_txt_dir = os.path.join(current_dir, "downloads_txt")
persistent_directory = os.path.join(current_dir, "db1", "chroma_db1")

# Ensure the txt directory exists
os.makedirs(downloads_txt_dir, exist_ok=True)

# Convert all PDFs to text files for plagiarism checking
def convert_pdfs_to_txt():
    for file in os.listdir(downloads_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(downloads_dir, file)
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(downloads_txt_dir, txt_filename)
            if not os.path.exists(txt_path):
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    content = " ".join([doc.page_content for doc in documents])
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(content)
                    print(f"✅ Converted {file} to {txt_filename}")
                except Exception as e:
                    print(f"❌ Error converting {file}: {e}")

# Convert PDFs on startup
convert_pdfs_to_txt()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize or load vectorstore
if not os.path.exists(persistent_directory) or not os.listdir(persistent_directory):
    print("⏳ Creating new vectorstore...")
    all_docs = []

    for file in os.listdir(downloads_dir):
        if file.endswith(".pdf"):
            path = os.path.join(downloads_dir, file)
            loader = PyPDFLoader(path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            all_docs.extend(split_docs)
            print(f"✅ Loaded {file}")

    if not all_docs:
        raise ValueError("No PDF files found in the downloads directory.")

    db = Chroma.from_documents(all_docs, embeddings, persist_directory=persistent_directory)
    print("✅ Vectorstore created.")
else:
    print("📂 Loading existing vectorstore...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Query function
def ask_question(query):
    results = db.similarity_search(query, k=3)
    if not results:
        return "❌ No relevant documents found."

    context_parts = []
    sources = set()
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
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
    if "❌ Answer not found" in answer:
        return answer
    else:
        source_list = ", ".join(sorted(sources))
        return f"{answer}\n\n📄 *Source(s)*: {source_list}"

# Preprocess text: lowercase and remove non-letter characters
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Generate k-length shingles
def get_shingles(text, k):
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        shingles.add(shingle)
    return shingles

# Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Rabin-Karp match count
def rabin_karp_count(pattern, text, prime=101):
    d = 256
    m = len(pattern)
    n = len(text)
    h = pow(d, m-1, prime)
    p = 0  # pattern hash
    t = 0  # text hash
    count = 0

    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    for i in range(n - m + 1):
        if p == t:
            if text[i:i+m] == pattern:
                count += 1
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % prime
            if t < 0:
                t += prime
    return count

# Function to run plagiarism check
def check_plagiarism(uploaded_file):
    if not uploaded_file.name.endswith(".txt"):
        return "❌ Please upload a .txt file."

    uploaded_path = os.path.join(downloads_txt_dir, os.path.basename(uploaded_file.name))
    with open(uploaded_file, 'rb') as src, open(uploaded_path, 'wb') as dst:
        dst.write(src.read())

    with open(uploaded_path, 'r', encoding='utf-8') as f:
        main_text = f.read()

    main_clean = preprocess(main_text)
    k = 5
    main_shingles = list(get_shingles(main_clean, k))
    sample_shingles = main_shingles[:500]  # Limit for Rabin-Karp

    if not sample_shingles:
        return "⚠️ Not enough content for plagiarism check."

    results = []
    combined_text = ""
    for filename in os.listdir(downloads_txt_dir):
        if filename.endswith(".txt") and filename != os.path.basename(uploaded_path):
            try:
                with open(os.path.join(downloads_txt_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    combined_text += " " + text  # for Jaccard
                    clean_text = preprocess(text)

                    # ✅ FIXED: count once per shingle if present
                    rk_matches = sum([1 for shingle in sample_shingles if rabin_karp_count(shingle, clean_text) > 0])
                    rk_score = rk_matches / len(sample_shingles)
                    results.append(f"   - {filename} → {rk_score:.2%}")
            except:
                continue

    # Overall Jaccard
    combined_clean = preprocess(combined_text)
    combined_shingles = get_shingles(combined_clean, k)
    jaccard = jaccard_similarity(set(main_shingles), combined_shingles)

    if not results:
        return "⚠️ No other .txt files found in the downloads_txt folder to compare."

    return (
        f"📄 Jaccard Similarity (Overall): {jaccard:.2%}\n"
        f"🧠 Rabin-Karp Match Ratios (per document):\n" +
        "\n".join(results)
    )

# Gradio UI
def launch_ui():
    with gr.Blocks() as demo:
        with gr.Tab("📘 Chat with PDFs"):
            gr.Markdown("# 📘 Multi-PDF Chatbot")
            chatbot = gr.Chatbot(label="Chat History")
            query_input = gr.Textbox(lines=2, placeholder="Ask something from the documents...")
            ask_button = gr.Button("Ask")
            history_state = gr.State([])

            def chat_with_memory(query, history):
                answer = ask_question(query)
                history.append((query, answer))
                return history, history, ""

            ask_button.click(
                fn=chat_with_memory,
                inputs=[query_input, history_state],
                outputs=[chatbot, history_state, query_input]
            )

        with gr.Tab("🧪 Plagiarism Checker"):
            gr.Markdown("## Upload a .txt file to compare against text extracted from PDFs")
            upload_file = gr.File(label="Upload .txt file")
            result = gr.Textbox(label="Similarity Score")
            check_button = gr.Button("Check Plagiarism")

            check_button.click(fn=check_plagiarism, inputs=upload_file, outputs=result)

    demo.launch()

if __name__ == "__main__":
    launch_ui()
