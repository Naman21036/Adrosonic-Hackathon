"""Central Loguru configuration with a standard-library logging bridge."""

from __future__ import annotations

import logging as std_logging
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import settings
from .context import get_correlation_fields, get_request_id

_configured = False
_SENSITIVE_FIELD_PARTS = ("api_key", "secret", "token", "authorization", "credential", "password")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_KEY_LIKE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b")


def _known_secrets() -> list[str]:
    values = []
    for name in (
        "azure_openai_api_key", "langfuse_public_key", "langfuse_secret_key",
        "mistral_api_key", "huggingface_api_key", "cloudflare_api_token",
    ):
        value = getattr(settings, name, None)
        if isinstance(value, str) and value:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact_text(value: str) -> str:
    result = value
    for secret in _known_secrets():
        result = result.replace(secret, "[REDACTED]")
    result = _BEARER.sub(r"\1[REDACTED]", result)
    return _KEY_LIKE.sub("[REDACTED]", result)


def safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-shaped fields and redact secret values in structured data."""
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                safe_key = str(key)
                if any(part in safe_key.lower() for part in _SENSITIVE_FIELD_PARTS):
                    continue
                output[safe_key] = clean(item)
            return output
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    return clean(fields)


class InterceptHandler(std_logging.Handler):
    """Forward existing stdlib log records through the single Loguru pipeline."""

    def emit(self, record: std_logging.LogRecord) -> None:
        try:
            level = record.levelname if record.levelname in {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "EXCEPTION"} else "INFO"
            message = redact_text(record.getMessage())
            fields = {
                "request_id": get_request_id(),
                "event": "STDLIB_LOG",
                "source_logger": record.name,
                "source_module": record.module,
                "source_function": record.funcName,
                "source_line": record.lineno,
            }
            if record.exc_info and record.exc_info[0]:
                fields["exception_type"] = record.exc_info[0].__name__
            safe=safe_fields(fields)
            logger.bind(
                request_id=safe.get("request_id") or "-",
                event="STDLIB_LOG",
                fields={key:value for key,value in safe.items() if key not in {"request_id","event"}},
            ).log(level if level != "CRITICAL" else "ERROR", message)
        except Exception:
            self.handleError(record)


def _safe_record(record: dict[str, Any]) -> bool:
    record["message"] = redact_text(record["message"])
    record["extra"] = safe_fields(record.get("extra", {}))
    # Do not serialize exception text/traceback, which can contain request data
    # or provider response bodies. Call sites log safe error types separately.
    record["exception"] = None
    return True


def configure_logging(*, force: bool = False, log_file: str | None = None, console: bool = True) -> None:
    """Configure one concise console sink and one rotating JSONL sink."""
    global _configured
    if _configured and not force:
        return
    level = settings.log_level.upper()
    if level not in {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    try:
        logger.level("EXCEPTION")
    except ValueError:
        logger.level("EXCEPTION", no=40, color="<red><bold>")
    logger.remove()
    formatter = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <9} | {name}:{function} | request_id={extra[request_id]} | event={message} | {extra[fields]}"
    if console:
        logger.add(sys.stderr, level=level, format=formatter, filter=_safe_record, backtrace=False, diagnose=False, colorize=False)
    destination = log_file or settings.log_file
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(path, level=level, serialize=True, rotation="20 MB", retention="14 days", encoding="utf-8", enqueue=False, filter=_safe_record, backtrace=False, diagnose=False)
    std_logging.basicConfig(handlers=[InterceptHandler()], level=getattr(std_logging, level, std_logging.INFO), force=True)
    _configured = True


def log_event(event: str, *, level: str = "INFO", **fields: Any) -> None:
    """Write a structured event with the current request ID attached."""
    safe = safe_fields({**get_correlation_fields(), **fields})
    request_id = get_request_id()
    logger.bind(request_id=request_id or "-", event=event, fields=safe).log(level, event)


configure_logging()
