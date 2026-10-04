"""Failure-isolated Langfuse tracing built on its legacy non-OTel client API."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any, Iterator

from ..config import settings
from .context import get_correlation_fields
from .logging import log_event

_client: Any | None = None
_client_lock = Lock()
_client_warning_emitted = False
_ACTIVE_PARENT: ContextVar[Any | None] = ContextVar("langfuse_active_parent", default=None)


class Observation:
    """Buffer updates and safely end a Langfuse trace/span/generation."""

    def __init__(self, target: Any | None, kind: str) -> None:
        self.target = target
        self.kind = kind
        self.fields: dict[str, Any] = {}

    @property
    def trace_id(self) -> str | None:
        if self.target is None:
            return None
        return getattr(self.target, "trace_id", None) or getattr(self.target, "id", None)

    def update(self, **fields: Any) -> "Observation":
        for key, value in fields.items():
            if key == "metadata" and isinstance(value, dict) and isinstance(self.fields.get(key), dict):
                self.fields[key] = {**self.fields[key], **value}
            else:
                self.fields[key] = value
        return self

    def _finish(self) -> None:
        if self.target is None:
            return
        try:
            if self.kind == "trace":
                self.target.update(**self.fields)
            else:
                self.target.end(**self.fields)
        except Exception as exc:
            log_event("LANGFUSE_OBSERVATION_END_FAILED", level="WARNING", exception_type=type(exc).__name__)


def langfuse_configured() -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_host
    )


def get_langfuse_client() -> Any | None:
    global _client, _client_warning_emitted
    if not settings.langfuse_enabled:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        if not _client_warning_emitted:
            log_event("LANGFUSE_DISABLED_MISSING_CREDENTIALS", level="WARNING")
            _client_warning_emitted = True
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                enabled=True,
            )
            return _client
        except Exception as exc:
            if not _client_warning_emitted:
                log_event("LANGFUSE_INITIALIZATION_FAILED", level="WARNING", exception_type=type(exc).__name__)
                _client_warning_emitted = True
            return None


@contextmanager
def trace(name: str, *, input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Observation]:
    """Create one root trace and propagate its parent to nested observations."""
    client = get_langfuse_client()
    if client is None:
        yield Observation(None, "trace")
        return
    safe_metadata = {**get_correlation_fields(), **(metadata or {})}
    try:
        target = client.trace(name=name, input=input, metadata=safe_metadata)
    except Exception as exc:
        log_event("LANGFUSE_TRACE_START_FAILED", level="WARNING", exception_type=type(exc).__name__)
        yield Observation(None, "trace")
        return
    handle = Observation(target, "trace")
    log_event("LANGFUSE_TRACE_STARTED",trace_id=handle.trace_id,name=name)
    token = _ACTIVE_PARENT.set(target)
    try:
        yield handle
    except BaseException as exc:
        handle.update(output={"status": "error", "error_type": type(exc).__name__})
        raise
    finally:
        handle._finish()
        _ACTIVE_PARENT.reset(token)


@contextmanager
def observation(
    name: str,
    *,
    kind: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Iterator[Observation]:
    """Create a child span/generation, or transparently no-op outside a trace."""
    parent = _ACTIVE_PARENT.get()
    if parent is None:
        yield Observation(None, kind)
        return
    client_method = getattr(parent, "generation" if kind == "generation" else "span", None)
    if client_method is None:
        yield Observation(None, kind)
        return
    fields: dict[str, Any] = {
        "name": name,
        "input": input,
        "metadata": {**get_correlation_fields(), **(metadata or {})},
    }
    if kind == "generation":
        fields["model"] = model
        fields["model_parameters"] = model_parameters or {}
    try:
        target = client_method(**fields)
    except Exception as exc:
        log_event("LANGFUSE_OBSERVATION_START_FAILED", level="WARNING", observation=name, exception_type=type(exc).__name__)
        yield Observation(None, kind)
        return
    handle = Observation(target, kind)
    token = _ACTIVE_PARENT.set(target)
    try:
        yield handle
    except BaseException as exc:
        handle.update(level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        handle._finish()
        _ACTIVE_PARENT.reset(token)


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        log_event("LANGFUSE_FLUSH_FAILED", level="WARNING", exception_type=type(exc).__name__)
