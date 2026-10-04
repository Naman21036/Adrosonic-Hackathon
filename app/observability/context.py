"""Request correlation context shared by logs and tracing spans."""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_REQUEST_ID: ContextVar[str | None] = ContextVar("adrosonic_request_id", default=None)
_CORRELATION: ContextVar[dict[str, str]] = ContextVar("adrosonic_observability_context", default={})
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def new_request_id(candidate: str | None = None) -> str:
    """Keep a safe caller-supplied correlation ID or create a UUID."""
    if candidate and _SAFE_REQUEST_ID.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def get_correlation_fields() -> dict[str, str]:
    fields = dict(_CORRELATION.get())
    request_id = get_request_id()
    if request_id:
        fields["request_id"] = request_id
    return fields


@contextmanager
def correlation_context(**fields: str) -> Iterator[None]:
    token = _CORRELATION.set({**_CORRELATION.get(), **{k: v for k, v in fields.items() if v}})
    try:
        yield
    finally:
        _CORRELATION.reset(token)


@contextmanager
def request_id_context(request_id: str) -> Iterator[None]:
    token = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)
