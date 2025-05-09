import os
from dotenv import load_dotenv
import gradio as gr

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
persistent_directory = os.path.join(current_dir, "db1", "chroma_db1")

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

    # Build context with source info
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

# List available PDF files
pdf_files = [
    os.path.join(downloads_dir, f)
    for f in os.listdir(downloads_dir)
    if f.lower().endswith(".pdf")
]

# Gradio UI with Chat History
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Multi-PDF Chatbot")
    gr.Markdown("Ask questions from the PDFs in the ⁠ downloads ⁠ folder.")

    chatbot = gr.Chatbot(label="Chat History")
    query_input = gr.Textbox(lines=2, placeholder="Ask something from the documents...")
    ask_button = gr.Button("Ask")
    history_state = gr.State([])  # To store chat history

    def chat_with_memory(query, history):
        answer = ask_question(query)
        history.append((query, answer))
        return history, history, ""

    ask_button.click(
        fn=chat_with_memory,
        inputs=[query_input, history_state],
        outputs=[chatbot, history_state, query_input]
    )


    gr.Markdown("### 📂 Available PDFs:")
    for pdf in pdf_files:
        gr.File(value=pdf, label=os.path.basename(pdf))

if __name__ == "__main__":
    demo.launch()