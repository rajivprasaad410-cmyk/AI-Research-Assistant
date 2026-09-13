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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

if "query_engine" not in st.session_state:
    try:
        st.session_state.query_engine = get_rag_query_engine(similarity_top_k=st.session_state.top_k)
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
                    num_chunks = index_pdf_document(tmp_path, original_filename=uploaded_file.name)
                    # Reinitialize query engine so the new vectors are immediately live
                    st.session_state.query_engine = get_rag_query_engine(similarity_top_k=st.session_state.top_k)
                    st.success(f"Indexed '{uploaded_file.name}' ({num_chunks} chunks) successfully!")
                except Exception as ex:
                    st.error(f"Indexing failed: {ex}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    st.markdown("---")
    st.header("⚙️ Retrieval Parameters")
    st.session_state.top_k = st.slider("Similarity Top-K", min_value=1, max_value=8, value=st.session_state.top_k)
    
    if st.button("Update Top-K", use_container_width=True):
        try:
            st.session_state.query_engine = get_rag_query_engine(similarity_top_k=st.session_state.top_k)
            st.toast(f"Retriever updated to Top-{st.session_state.top_k} Chunks!")
        except Exception as e:
            st.error(f"Failed to update query engine: {e}")

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
                    st.caption(src["text"])

# Chat input handling
if prompt := st.chat_input("Ask a question about the papers (e.g., Explain Multi-Head Attention)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vectors & generating answer..."):
            try:
                # Ensure engine is instantiated
                if st.session_state.query_engine is None:
                    st.session_state.query_engine = get_rag_query_engine(similarity_top_k=st.session_state.top_k)

                response = st.session_state.query_engine.query(prompt)

                # Robust answer extraction
                answer_text = str(response).strip()
                if not answer_text or answer_text == "Empty Response":
                    if hasattr(response, "response") and response.response:
                        answer_text = response.response.strip()
                    else:
                        answer_text = "I could not find information on this topic in the indexed documents."

                st.markdown(answer_text)

                sources_data = []
                if hasattr(response, "source_nodes") and response.source_nodes:
                    with st.expander("🔍 Retrieved Citations & Passages"):
                        for idx, node in enumerate(response.source_nodes, 1):
                            fname = node.metadata.get("file_name", "Paper")
                            page = node.metadata.get("page_number", "?")
                            src_label = f"{fname} (Page {page})"
                            score = node.score if node.score is not None else 0.0
                            snippet = node.node.get_text().strip()
                            truncated_snippet = snippet[:400] + "..." if len(snippet) > 400 else snippet

                            sources_data.append({
                                "source": src_label,
                                "score": score,
                                "text": truncated_snippet
                            })

                            st.markdown(f"**[{idx}] {src_label}** — Score: `{score:.4f}`")
                            st.caption(truncated_snippet)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources_data
                })

            except Exception as err:
                st.error(f"Error generating answer: {err}")
