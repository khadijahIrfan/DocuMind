import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

import answer as rag

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="DocuMind | Study Assistant", page_icon="📘", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#14213d,#244b74); color: white; margin-bottom: 1rem;}
.hero h1 {color: white;}
.hero p {color: #e6eef8; margin-bottom: 0;}
div[data-testid="stMetric"] {background: var(--secondary-background-color); padding: 12px 16px; border-radius: 12px;}
</style>
<div class="hero">
<h1>📘 DocuMind</h1>
<p>Your AI study assistant for asking questions from your indexed PDF notes, with grounded answers and page citations.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading DocuMind resources…")
def load_resources():
    embeddings_file = BASE_DIR / "output" / "embeddings.npy"
    metadata_file = BASE_DIR / "output" / "embedding_metadata.json"
    if not embeddings_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Embeddings are missing. Run your existing extraction/chunking/embedding steps first.")

    embeddings = np.load(embeddings_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match. Run `python embed_text.py` again.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found. Check the .env file in E:\\DocuMind.")

    model = SentenceTransformer(rag.EMBEDDING_MODEL)
    client = genai.Client(api_key=api_key)
    return model, embeddings, metadata, client


with st.sidebar:
    st.title("DocuMind")
    st.caption("PDF Study Assistant")
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. Enter a question about your indexed notes.\n2. DocuMind finds relevant passages.\n3. Review the answer and source pages.")
    st.info("This version uses the PDF notes already indexed in your project.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Answer model: {rag.GENERATION_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Preparing the embedding model and notes…"):
        model, embeddings, metadata, client = load_resources()
except Exception as exc:
    st.error(f"DocuMind could not start: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Indexed passages", f"{len(metadata):,}")
c2.metric("Embedding dimensions", f"{embeddings.shape[1]:,}")
c3.metric("Answer model", rag.GENERATION_MODEL)

st.subheader("Ask your notes")
st.write("Try: *What is email filtering in NLP?*")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved source passages"):
                for source in message["sources"]:
                    pages = ", ".join(map(str, source.get("source_pages", []))) or "Unknown"
                    st.markdown(f"**Chunk {source.get('chunk_id', '—')} · Page(s): {pages}**  \nSemantic: {source.get('_semantic_score', 0):.3f} · Retrieval: {source.get('_retrieval_score', 0):.3f}")
                    st.write(source["text"])
                    st.divider()
        for warning in message.get("warnings", []):
            st.warning(warning)

question = st.chat_input("Ask a question about your NLP notes…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes…"):
                passages = rag.retrieve_passages(question, model, embeddings, metadata)

            if not passages:
                response = ("I couldn't find sufficiently relevant passages to answer confidently from your notes. "
                            "Try rephrasing the question or asking about a topic covered in the PDF.")
                st.info(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
            else:
                with st.spinner("Generating an answer with Gemini…"):
                    response = rag.generate_answer(client, question, passages)
                st.markdown(response)
                warnings = rag.validate_citations(response, passages)
                for warning in warnings:
                    st.warning(warning)

                pages = sorted({p for passage in passages for p in passage.get("source_pages", [])})
                if pages:
                    st.caption("Source pages: " + ", ".join(map(str, pages)))

                with st.expander("View retrieved source passages"):
                    for source in passages:
                        source_pages = ", ".join(map(str, source.get("source_pages", []))) or "Unknown"
                        st.markdown(f"**Chunk {source.get('chunk_id', '—')} · Page(s): {source_pages}**  \nSemantic: {source.get('_semantic_score', 0):.3f} · Retrieval: {source.get('_retrieval_score', 0):.3f}")
                        st.write(source["text"])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant", "content": response, "sources": passages, "warnings": warnings
                })
        except Exception as exc:
            error_message = f"Something went wrong while answering: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
