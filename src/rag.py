import os
import sys
from typing import Optional

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from src.indexer import get_embedding_model, DB_DIR, COLLECTION_NAME


def get_groq_api_key() -> str:
    """Retrieves the Groq API key from Streamlit Cloud Secrets or local .env,"""
    # 1. Check Streamlit secrets first (Streamlit Cloud deployment)
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip().strip('"').strip("'")
            if key:
                return key
    except Exception:
        pass

    # 2. Check local environment variable (.env)
    env_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if env_key:
        return env_key

    return ""


def get_rag_query_engine(similarity_top_k: int = 3):
    """Initializes and returns a production-grade LlamaIndex QueryEngine

    connected to ChromaDB with precise citation prompting.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please add it to Streamlit Secrets or your local .env file."
        )

    # Set in os.environ for internal SDK compatibility
    os.environ["GROQ_API_KEY"] = api_key

    # Initialize Groq LLM
    llm = Groq(
        model="qwen/qwen3.6-27b",
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    # Initialize ChromaDB persistent connection
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    qa_template = PromptTemplate(
        "You are an expert academic research assistant.\n"
        "Answer the question thoroughly and accurately based ONLY on the provided context.\n"
        "Always cite source documents and page numbers where applicable.\n"
        "If the answer cannot be found in the context, explicitly state that the document does not mention it.\n\n"
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

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


def ask_question(query: str):
    """Utility function to query the RAG pipeline directly."""
    query_engine = get_rag_query_engine()
    response = query_engine.query(query)
    return response


if __name__ == "__main__":
    test_query = "What is the key mechanism introduced in the Attention paper?"
    print(f"Query: {test_query}\n")
    try:
        res = ask_question(test_query)
        print("Response:\n", res)
    except Exception as e:
        print(f"Error: {e}")