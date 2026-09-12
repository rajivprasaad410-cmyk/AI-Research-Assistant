import os
import sys

# Ensure root directory is in python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
import chromadb

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from src.indexer import get_embedding_model, DB_DIR, COLLECTION_NAME

# Load GROQ_API_KEY from .env
load_dotenv()

# Prompt enforcing inline page citations
CITATION_PROMPT = PromptTemplate(
    "You are an expert scientific research assistant.\n"
    "Context information from the research paper is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Always cite the exact page number(s) where information was obtained "
    "using the format: [Page X]. If multiple pages are used, cite all of them.\n"
    "If the answer cannot be determined from the context, state that clearly.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

def get_rag_query_engine(similarity_top_k: int = 3):
    """
    Loads the persistent ChromaDB index, binds the Groq LLM,
    and returns a query engine ready for questions.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "paste_your_groq_api_key_here":
        raise ValueError("Please provide a valid GROQ_API_KEY in your .env file.")

    # 1. Reconnect to the local ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = get_embedding_model()

    # 2. Reconstruct index from existing vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    # 3. Configure Groq LLM using the supported free-tier model
    llm = Groq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.1
    )

    # 4. Build query engine with custom prompt
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        text_qa_template=CITATION_PROMPT
    )

    return query_engine

if __name__ == "__main__":
    engine = get_rag_query_engine()
    test_query = "What is the Scaled Dot-Product Attention formula and why is it scaled?"
    print(f"\nQuerying: '{test_query}'...\n")
    
    response = engine.query(test_query)
    print("--- Response ---")
    print(response.response)
    
    print("\n--- Retrieved Source Chunks ---")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"[{i}] {node.metadata.get('source')} | Similarity Score: {node.score:.4f}")