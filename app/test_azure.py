"""Send one small request to verify Azure OpenAI configuration."""
from __future__ import annotations

import asyncio
import sys

from .azure_openai import AzureOpenAIError, test_connection as _test_connection


def main() -> int:
    try:
        print(asyncio.run(_test_connection()))
        return 0
    except AzureOpenAIError as exc:
        print(f"Azure OpenAI test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
