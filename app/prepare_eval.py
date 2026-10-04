"""Extract up to N MS MARCO queries with their provided answers and selected passages."""
import argparse,json
from pathlib import Path
from .config import settings
from .datasets import load_split

def _local_parquet_rows(path: str):
    """Stream rows from an already downloaded parquet file without HF cache writes."""
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=20); p.add_argument("--output",default="reports/phase2/evaluation/evaluation_dataset.jsonl"); p.add_argument("--local-file",help="Use an existing local validation parquet file without contacting Hugging Face"); p.add_argument("--source-config",default=None,help="Dataset config/version represented by --local-file"); a=p.parse_args()
    rows=_local_parquet_rows(a.local_file) if a.local_file else load_split("validation")
    source={"dataset":settings.dataset_name,"config":a.source_config or settings.dataset_config,"split":"validation","local_file":str(Path(a.local_file).resolve()) if a.local_file else None}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); n=0
    with out.open("w",encoding="utf-8") as f:
        for row in rows:
            q=(row.get("query") or "").strip(); answers=row.get("answers",[]); passages=row.get("passages",{}); texts=passages.get("passage_text",[]) if isinstance(passages,dict) else []; chosen=passages.get("is_selected",[]) if isinstance(passages,dict) else []
            relevant=[t for t,flag in zip(texts,chosen) if flag and t.strip()]
            if q and answers and relevant:
                f.write(json.dumps({"query_id":str(row.get("query_id",n)),"query":q,"reference":" ".join(answers) if isinstance(answers,list) else str(answers),"relevant_contexts":relevant},ensure_ascii=False)+"\n"); n+=1
            if n>=a.count: break
    manifest={"source":source,"rows_written":n,"format":"MS MARCO query, answer, and selected validation passages","output":str(out)}
    out.with_name("evaluation_dataset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"saved_valid_queries={n} file={out}")
if __name__=="__main__": main()
