"""Streamlit demonstration client."""
import streamlit as st
import requests
from adrosonic_retrieval.config import get_settings
settings = get_settings()
st.set_page_config(page_title="Adrosonic Dense Search", page_icon="🔎", layout="wide")
st.title("MS MARCO Dense Passage Search")
st.caption("Phase 1 baseline · Hugging Face embeddings · local Qdrant · top 5")
api = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")
try:
    health = requests.get(f"{api.rstrip('/')}/health", timeout=3).json()
    st.sidebar.success(f"API: {health['status']} · passages: {health.get('indexed_passages')}")
except requests.RequestException as exc:
    st.sidebar.error(f"API unavailable: {exc}")
query = st.text_area("Search query", placeholder="What is the role of the mitochondria?")
if st.button("Search", type="primary", disabled=not query.strip()):
    try:
        response = requests.post(f"{api.rstrip('/')}/search", json={"query": query, "top_k": 5}, timeout=60)
        if response.status_code >= 400:
            st.error(response.json().get("detail", response.text))
        else:
            payload = response.json()
            st.caption(f"Mode: {payload['retrieval_mode']} · server latency: {payload['latency_ms']:.1f} ms")
            for rank, item in enumerate(payload["results"], 1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {item['passage_id']}** · cosine: `{item['score']:.4f}` · `{item['retrieval_mode']}`")
                    st.write(item["text"])
                    st.caption(f"Source: {item['source']}")
    except requests.RequestException as exc:
        st.error(f"Search request failed: {exc}")
