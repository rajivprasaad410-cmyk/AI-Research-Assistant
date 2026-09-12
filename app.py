import os
import tempfile
import streamlit as st

from src.indexer import index_pdf_document
from src.rag import get_rag_query_engine

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Academic Research Assistant")
st.caption("Layout-Aware RAG with LlamaIndex, ChromaDB, BGE Embeddings & Groq")

# Initialize persistent session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    try:
        st.session_state.query_engine = get_rag_query_engine(similarity_top_k=3)
    except Exception:
        st.session_state.query_engine = None

# Sidebar controls
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload an academic PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Document into ChromaDB", use_container_width=True):
            with st.spinner(f"Extracting layout and embedding '{uploaded_file.name}'..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # Pass the original file name so citations stay accurate
                    index_pdf_document(tmp_path, original_filename=uploaded_file.name)
                    # Reinitialize query engine with updated vectors
                    st.session_state.query_engine = get_rag_query_engine(similarity_top_k=3)
                    st.success(f"Indexed '{uploaded_file.name}' successfully!")
                except Exception as ex:
                    st.error(f"Indexing failed: {ex}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.markdown("---")
    st.header("⚙️ Retrieval Parameters")
    top_k = st.slider("Similarity Top-K", min_value=1, max_value=6, value=3)
    
    if st.button("Update Top-K", use_container_width=True):
        if st.session_state.query_engine is not None:
            st.session_state.query_engine = get_rag_query_engine(similarity_top_k=top_k)
            st.toast(f"Retriever updated to Top-{top_k} Chunks!")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Retrieved Citations & Passages"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{idx}] {src['source']}** — Similarity: `{src['score']:.4f}`")
                    st.caption(src["text"][:350] + "..." if len(src["text"]) > 350 else src["text"])

# Chat input handling
if prompt := st.chat_input("Ask a question about the papers (e.g., Explain Multi-Head Attention)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.query_engine is None:
        with st.chat_message("assistant"):
            st.error("Vector index is not initialized. Please ensure documents are indexed.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching vectors & generating answer..."):
                try:
                    response = st.session_state.query_engine.query(prompt)
                    answer_text = response.response
                    st.markdown(answer_text)

                    sources_data = []
                    if hasattr(response, "source_nodes") and response.source_nodes:
                        with st.expander("🔍 Retrieved Citations & Passages"):
                            for idx, node in enumerate(response.source_nodes, 1):
                                src_label = node.metadata.get("source", "Unknown Page")
                                score = node.score if node.score is not None else 0.0
                                snippet = node.node.get_text().strip()
                                
                                sources_data.append({
                                    "source": src_label,
                                    "score": score,
                                    "text": snippet
                                })
                                
                                st.markdown(f"**[{idx}] {src_label}** — Similarity: `{score:.4f}`")
                                st.caption(snippet[:350] + "..." if len(snippet) > 350 else snippet)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data
                    })
                except Exception as err:
                    st.error(f"Error generating answer: {err}")