# Script to ingest the pdf into the vector db

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Load PDF and extract text
def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())


# Split text into chunks
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# Store chunks in ChromaDB
def ingest(pdf_path: str = "manual.pdf"):
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks created")

    print("Connecting to ChromaDB...")
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_store")
    collection = client.get_or_create_collection(
        name="maintaince_docs",
        embedding_function=embedder
    )

    print("Embedding and storing chunks...")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"  → Ingestion complete. {len(chunks)} chunks stored.")


if __name__ == "__main__":
    ingest()