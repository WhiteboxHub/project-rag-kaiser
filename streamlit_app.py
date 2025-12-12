import streamlit as st
import os
import shutil
from rag.query_pipeline import RAGPipeline
from app.ingestion.embedder import Embedder

# Page config
st.set_page_config(
    page_title="Kaiser RAG POC",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Kaiser Policy RAG Assistant")
st.markdown("Ask questions about Kaiser health insurance policies and member guides.")

# Initialize RAG Pipeline
# We cache this to avoid reloading the model on every interaction
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

try:
    rag = get_rag_pipeline()
    st.success("RAG Pipeline Loaded Successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load RAG Pipeline: {e}")
    st.stop()

# Sidebar for Admin/Debug
with st.sidebar:
    st.header("Admin Controls")
    
    if st.button("Clear Vector Database", type="secondary"):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/embeddings/chroma")
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                st.warning("Vector Database cleared. Please re-run ingestion.")
                # Clear resource cache to force reload if needed, though mostly affects retriever
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error clearing DB: {e}")
        else:
            st.info("No database found to clear.")

    st.divider()
    st.info("Model: sentence-transformers/all-MiniLM-L6-v2")


# Main chat interface
query = st.text_input("Enter your question:", placeholder="e.g., What are the benefits of the Gold plan?")

if query:
    with st.spinner("Searching policies and generating answer..."):
        try:
            result = rag.query(query)
            
            # Display Answer
            st.markdown("### Answer")
            st.markdown(result["answer"])
            
            # Display Context with Metadata
            with st.expander("View Retrieved Context Sources"):
                metadatas = result.get("metadata", [])
                for i, (chunk, score) in enumerate(zip(result["context"], result["scores"])):
                    # Get metadata for this chunk
                    meta = metadatas[i] if i < len(metadatas) else {}
                    
                    # Build metadata display
                    meta_str = ""
                    if meta.get("source_file"):
                        meta_str += f"📄 **{meta['source_file']}**"
                    if meta.get("page"):
                        meta_str += f" | 📖 Page {meta['page']}"
                    if meta.get("chapter"):
                        meta_str += f" | 📑 Chapter {meta['chapter']}"
                    
                    st.markdown(f"**Source {i+1}** (Relevance: {score:.3f})")
                    if meta_str:
                        st.caption(meta_str)
                    st.info(chunk)
                    st.divider()
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("POC - Kaiser RAG Project")
