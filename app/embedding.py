from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


def resolve_device(torch, requested: str) -> str:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("EMBEDDING_DEVICE=cuda was requested, but this PyTorch build cannot access CUDA")
    return "cuda" if requested == "auto" and cuda_available else ("cpu" if requested == "auto" else requested)


class Embedder(Protocol):
    dimension: int
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class QwenEmbedder:
    """Load Qwen3 once; use CUDA FP16 when available and adapt on CUDA OOM."""
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "auto",
        token: str | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
        precision: str | None = None,
        direct_device_map: bool = False,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        cuda_available = torch.cuda.is_available()
        self.device = resolve_device(torch, device)
        if self.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA device selected but torch.cuda.is_available() is false")
        requested_precision = (precision or settings.embedding_precision).lower()
        if self.device == "cuda":
            if requested_precision == "float16" or requested_precision == "auto":
                self.torch_dtype = torch.float16
            elif requested_precision == "bfloat16" and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            elif requested_precision == "float32":
                self.torch_dtype = torch.float32
            else:
                logger.warning("Requested %s is unsupported on CUDA; using float16", requested_precision)
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
            if requested_precision not in {"auto", "float32"}:
                logger.info("CPU inference uses float32; ignoring requested %s precision", requested_precision)
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_seq_length = max_seq_length or settings.embedding_max_seq_length
        if self.batch_size < 1 or self.max_seq_length < 8 or self.max_seq_length > 512:
            raise ValueError("batch_size must be positive and max_seq_length must be between 8 and 512")
        self.model_name = model_name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token or None)
        if direct_device_map and self.device != "cuda":
            raise ValueError("direct_device_map is only supported when the embedder device is CUDA")
        model_kwargs = {
            "revision": revision,
            "token": token or None,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.torch_dtype,
        }
        if direct_device_map:
            # Let Transformers/Accelerate stream checkpoint tensors onto CUDA
            # instead of first materializing the full model in host memory.
            model_kwargs["device_map"] = {"": 0}
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if not direct_device_map:
            self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda" and next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("CUDA was selected but the embedding model parameters are not on the GPU")
        self.dimension = int(self.model.config.hidden_size)
        if self.dimension != 1024:
            raise RuntimeError(f"Expected Qwen3-Embedding-0.6B dimension 1024, received {self.dimension}")
        self.torch = torch
        self.effective_precision = str(self.torch_dtype).removeprefix("torch.")
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        logger.info(
            "Embedding model=%s revision=%s device=%s gpu=%s precision=%s batch_size=%d max_seq_length=%d dimension=%d",
            model_name, revision, self.device, self.gpu_name or "n/a", self.effective_precision,
            self.batch_size, self.max_seq_length, self.dimension,
        )

    def _encode_batch(self, texts: list[str], query: bool) -> np.ndarray:
        prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
        inputs = [prefix + t if query else t for t in texts]
        try:
            batch = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            if self.device == "cuda" and any(value.device.type != "cuda" for value in batch.values()):
                raise RuntimeError("Tokenized inputs were not placed on the selected CUDA device")
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                positions = self.torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand_as(batch["attention_mask"])
                last_positions = positions.masked_fill(batch["attention_mask"] == 0, -1).max(dim=1).values
                vectors = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), last_positions]
                vectors = self.torch.nn.functional.normalize(vectors.to(dtype=self.torch.float32), p=2, dim=1)
                vectors = vectors.to(dtype=self.torch.float32).contiguous()
            if vectors.device.type != self.device:
                raise RuntimeError(f"Embedding output is on {vectors.device}, expected {self.device}")
            # Synchronize before host conversion so asynchronous CUDA faults surface
            # at the operation that produced the vectors. Set CUDA_LAUNCH_BLOCKING=1
            # in the shell for more precise kernel-level diagnostics.
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            result = vectors.cpu().numpy()
            if result.shape != (len(texts), self.dimension):
                raise RuntimeError(f"Unexpected embedding shape {result.shape}; expected {(len(texts), self.dimension)}")
            if not np.isfinite(result).all():
                raise RuntimeError("Embedding output contains NaN or infinite values")
            return result
        except self.torch.cuda.OutOfMemoryError:
            if self.device != "cuda" or len(texts) <= 1:
                raise
            self.torch.cuda.empty_cache()
            logger.warning("CUDA out of memory for batch=%d; retrying as two smaller batches", len(texts))
            midpoint = len(texts) // 2
            self.batch_size = midpoint
            return np.concatenate((self._encode_batch(texts[:midpoint], query), self._encode_batch(texts[midpoint:], query)))
        except Exception as exc:
            message = str(exc).lower()
            if self.device == "cuda" and ("illegal memory access" in message or "device-side assert" in message):
                logger.exception(
                    "CUDA kernel failure during embedding; no CPU fallback was attempted. "
                    "Retry in a fresh process with CUDA_LAUNCH_BLOCKING=1 to locate the failing operation."
                )
                raise RuntimeError(
                    "CUDA embedding failed with an illegal memory access. No CPU fallback was attempted; "
                    "restart the process and diagnose with CUDA_LAUNCH_BLOCKING=1."
                ) from exc
            raise

    def _encode(self, texts: list[str], query: bool, batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        active_batch = batch_size or self.batch_size
        chunks = []
        start = 0
        while start < len(texts):
            chunks.append(self._encode_batch(texts[start:start + active_batch], query))
            start += min(active_batch, len(texts) - start)
            active_batch = min(active_batch, self.batch_size)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, query=False, batch_size=batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], query=True)[0]


class HashEmbedder:
    """Deterministic tiny fixture embedder; never used as a production default."""
    def __init__(self, dimension: int = 128): self.dimension = dimension
    def encode_documents(self, texts: list[str]) -> np.ndarray: return np.stack([self.encode_query(t) for t in texts])
    def encode_query(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            h = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            vec[idx] += 1 if h[4] & 1 else -1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec
