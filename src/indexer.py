import os
import sys
from typing import List

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import chromadb
import nltk

# Fix for NLTK hardlink security check (CWE-59) on Linux cloud environments
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.parser import parse_pdf_to_documents

# Absolute path guarantees ChromaDB reads and writes to the exact same folder
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "academic_research_papers"


def get_embedding_model() -> HuggingFaceEmbedding:
    """Initializes BAAI/bge-small-en-v1.5 embedding model."""
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        trust_remote_code=True
    )


def index_pdf_document(pdf_path: str, original_filename: str = None):
    """
    Parses PDF into layout-aware Markdown nodes and persists them in ChromaDB.
    """
    display_name = original_filename or os.path.basename(pdf_path)
    print(f"1. Parsing PDF: {display_name}")

    # Layout-aware extraction via PyMuPDF4LLM
    documents = parse_pdf_to_documents(pdf_path, original_filename=display_name)
    if not documents:
        raise ValueError(f"No extractable text found in {display_name}.")

    print("2. Connecting to ChromaDB...")
    os.makedirs(DB_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Delete existing entries for this file to prevent duplicate vectors
    try:
        chroma_collection.delete(where={"file_name": display_name})
        print(f"Cleared prior records for: {display_name}")
    except Exception:
        pass

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = get_embedding_model()
    Settings.embed_model = embed_model

    # Recursive token chunker (512 token ceiling to match bge-small-en-v1.5)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"3. Embedding and storing {len(nodes)} chunks...")
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print(f"Indexing complete! Vectors saved to {DB_DIR}")
    return len(nodes)
