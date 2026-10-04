"""Run a hybrid retrieval query and generate a grounded Azure OpenAI answer."""
from __future__ import annotations

import argparse
import asyncio
import sys

from .azure_openai import AzureOpenAIError, grounded_answer
from .config import settings
from .embedding import QwenEmbedder
from .retrieval import search
from .storage import HybridIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve passages and answer with Azure OpenAI")
    parser.add_argument("query", help="Question to search for")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("query must not be empty")

    index = None
    try:
        embedder = QwenEmbedder(
            settings.hf_model_name,
            settings.hf_model_revision,
            settings.device,
            settings.huggingface_api_key,
        )
        index = HybridIndex(embedder)
        result = search(index, args.query, mode="hybrid", top_k=settings.top_k)
        answer = asyncio.run(grounded_answer(args.query, result["results"]))
    except AzureOpenAIError as exc:
        print(f"Azure OpenAI error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"RAG query failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()

    print("Query:")
    print(args.query)
    print("\nRetrieved Passages:")
    for number, passage in enumerate(result["results"], start=1):
        print(f"{number}. [{passage['id']}] {passage['text']}")
    print("\nAnswer:")
    print(answer["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
