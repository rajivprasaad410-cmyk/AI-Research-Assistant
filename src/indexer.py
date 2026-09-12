import os
import sys
from typing import Optional

# Ensure project root is in the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.parser import parse_pdf_to_documents

DB_DIR = "./chroma_db"
COLLECTION_NAME = "research_papers"

def get_embedding_model():
    """Initializes a lightweight, fast, local embedding model (runs 100% free)."""
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def index_pdf_document(pdf_path: str, original_filename: Optional[str] = None):
    """
    Parses a PDF, chunks it, converts it into vector embeddings,
    and persists it into a local ChromaDB database.
    Automatically purges prior records for the same document.
    """
    file_name = original_filename if original_filename else os.path.basename(pdf_path)
    
    print(f"1. Parsing {pdf_path} (Document: {file_name})...")
    documents = parse_pdf_to_documents(pdf_path, original_filename=file_name)

    print("2. Initializing ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Automatically purge existing entries for this specific document
    try:
        chroma_collection.delete(where={"file_name": file_name})
        print(f"Cleared prior records for '{file_name}'.")
    except Exception:
        pass

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embedding_model()

    # Split text into 512-token chunks with 50-token overlap to maintain context
    transformations = [
        SentenceSplitter(chunk_size=512, chunk_overlap=50)
    ]

    print("3. Generating embeddings and storing vectors...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=transformations,
        show_progress=True
    )

    print(f"Indexing complete! Vectors saved locally in '{DB_DIR}'.")
    return index

if __name__ == "__main__":
    test_pdf = os.path.join("data", "attention.pdf")
    index_pdf_document(test_pdf)