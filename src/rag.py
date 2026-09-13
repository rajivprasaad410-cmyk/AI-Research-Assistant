import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from src.indexer import get_embedding_model, DB_DIR, COLLECTION_NAME


def get_groq_api_key() -> str:
    """Retrieves Groq API key from Streamlit Secrets or environment."""
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'")
            if key:
                return key
    except Exception:
        pass

    env_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if env_key:
        return env_key

    return ""


def get_rag_query_engine(similarity_top_k: int = 3):
    """
    Initializes and returns a retriever query engine connected to ChromaDB.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Configure it in secrets or .env.")

    os.environ["GROQ_API_KEY"] = api_key

    # Production LLM on Groq (no reasoning-token exhaustion)
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    )
    Settings.llm = llm

    embed_model = get_embedding_model()
    Settings.embed_model = embed_model

    # Connect to persistent ChromaDB storage
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    qa_template = PromptTemplate(
        "You are an expert academic research assistant.\n"
        "Answer the question thoroughly and accurately based ONLY on the provided context.\n"
        "Cite the document name and page number for facts wherever possible.\n"
        "If the answer cannot be found in the context, clearly state that the provided papers do not contain the answer.\n\n"
        "Context Information:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_template,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
