import json
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(page_title="Adrosonic | Precision Retrieval",page_icon="🔎",layout="wide")
st.title("🔎 Precision Retrieval for RAG")
st.caption("Phase 1 dense baseline · Phase 2 BM25 + Qdrant hybrid search")
api=st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
try:
    health=requests.get(api+"/health",timeout=2).json(); status=requests.get(api+"/status",timeout=2).json()
    st.sidebar.success("Retrieval ready" if health["status"]=="ready" else "Service unavailable")
    st.sidebar.metric("Indexed passages",f"{status.get('indexed_passages',0):,}")
except Exception as e:
    st.sidebar.error(f"API unavailable: {e}"); health={}

query=st.text_input("Ask a question",placeholder="What are the main causes of ...")
generate=st.checkbox("Generate a grounded Mistral answer (requires confirmed free-tier allowance)",disabled=not health.get("mistral_ready",False))
with st.expander("Metadata filters"):
    filter_by_source=st.checkbox("Filter before retrieval by source_id",value=False)
    source_id=st.number_input("Source ID",min_value=0,value=19699,step=1,disabled=not filter_by_source)
filters={"source_id":int(source_id)} if filter_by_source else {}
c1,c2,c3=st.columns(3)
if c1.button("Search dense baseline",type="primary",disabled=not query): mode="dense"
elif c2.button("Search hybrid",disabled=not query): mode="hybrid"
elif c3.button("Compare both",disabled=not query): mode="compare"
else: mode=None
if mode:
    try:
        modes=("dense","hybrid") if mode=="compare" else (mode,)
        cols=st.columns(len(modes))
        for col,current in zip(cols,modes):
            response=requests.post(api+"/search",json={"query":query,"mode":current,"top_k":5,"filter":filters,"generate_answer":generate and current=="hybrid"},timeout=120); response.raise_for_status(); data=response.json()
            with col:
                if data.get("filter"):
                    st.info(f"Pre-retrieval filter: source_id = {data['filter']['source_id']}")
                else:
                    st.caption("Filter: none (searching the eligible corpus)")
                st.subheader(f"{current.title()} · {data.get('latency_ms',0):.1f} ms")
                if "answer" in data: st.markdown(data["answer"]["answer"]); st.caption(f"Generated in {data['answer']['generation_latency_ms']:.1f} ms · cited: {', '.join(data['answer']['supporting_passage_ids'])}")
                for i,hit in enumerate(data.get("results",[]),1):
                    with st.container(border=True):
                        st.markdown(f"**{i}. {hit['id']}** · ranking score `{hit['score']:.5f}` · `{hit['retrieval_mode']}`")
                        st.write(hit["text"]); st.caption(f"source_id: {hit.get('source_id')} · {json.dumps(hit.get('metadata',{}),ensure_ascii=False)}")
    except Exception as e: st.error(f"Search failed: {e}")
st.divider()
left,right=st.columns(2)
with left:
    st.subheader("Evaluation results")
    p=Path("reports/phase2/evaluation/evaluation_results.json")
    if p.exists():
        ev=json.loads(p.read_text(encoding="utf-8")); st.write(ev.get("aggregate",{})); st.caption(ev.get("evaluation_error") or f"{ev.get('query_count')} queries · {ev.get('run_timestamp_utc')}")
    else: st.info("No actual evaluation artifact yet. Run `python -m app.evaluate`.")
with right:
    st.subheader("Latency benchmark")
    p=Path("reports/phase2/benchmarks/benchmark_summary.json")
    if p.exists(): st.json(json.loads(p.read_text(encoding="utf-8")))
    else: st.info("No actual benchmark summary yet. Run `python -m app.benchmark`.")
st.subheader("Indexing report")
p=Path("reports/phase2/indexing_run.json")
if p.exists(): st.json(json.loads(p.read_text(encoding="utf-8")))
else: st.info("No indexing run recorded yet. Run `python -m app.ingest --limit 100000`.")
st.subheader("Index maintenance")
with st.expander("Upsert a passage"):
    with st.form("upsert"):
        pid=st.text_input("Passage ID"); text=st.text_area("Passage text"); meta_source=st.text_input("Source metadata",value="demo")
        ok=st.form_submit_button("Upsert")
    if ok:
        r=requests.post(api+"/upsert",json={"passage_id":pid,"text":text,"metadata":{"source":meta_source}},timeout=120); r.raise_for_status(); st.success(r.json())
with st.expander("Delete a passage"):
    with st.form("delete"):
        delete_id=st.text_input("Passage ID to delete"); confirm=st.checkbox("Confirm deletion")
        go=st.form_submit_button("Delete",disabled=not confirm)
    if go:
        r=requests.delete(api+"/passages/"+delete_id,timeout=60); r.raise_for_status(); st.success(r.json())
