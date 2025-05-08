import os
from dotenv import load_dotenv
import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Load environment variables
load_dotenv()

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "BCI.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load or initialize vector store
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Query function
def ask_question(query):
    relevant_docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content

books_dir = os.path.join(current_dir, "books")
pdf_files = [
    os.path.join(books_dir, f)
    for f in os.listdir(books_dir)
    if f.lower().endswith(".pdf")
]
downloadables = [gr.File(value=pdf, label=os.path.basename(pdf)) for pdf in pdf_files]
# Gradio Interface
iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about the BCI.pdf..."),
    outputs=["text"] + downloadables,
    title="Research Buddies",
    description="Ask a question related to your field of research (Data Science)."
)


if __name__ == "__main__":
    iface.launch()
