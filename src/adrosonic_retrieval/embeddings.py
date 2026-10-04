"""Sentence-transformers wrapper with normalized query/document vectors."""
from collections.abc import Sequence
import logging
import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", *, dimension: int = 0,
                 precision: str = "fp32", query_instruction: str = "",
                 token: str | None = None, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.requested_dimension = dimension
        self.requested_device = device
        self.device = device
        self.precision = precision.lower()
        qwen = model_name.lower().startswith("qwen/")
        kwargs = {}
        if token:
            kwargs["token"] = token
        if revision:
            kwargs["revision"] = revision
        if dimension:
            if qwen and not 32 <= dimension <= 1024:
                raise ValueError("Qwen3-Embedding-0.6B supports output dimensions from 32 through 1024")
            kwargs["truncate_dim"] = dimension
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        effective_precision = self.precision
        if self.device.lower().startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA was requested but is unavailable; falling back to CPU FP32")
                self.device = "cpu"
        if self.device.lower() == "cpu" and effective_precision != "fp32":
            log.warning("CPU inference uses FP32; ignoring requested %s precision", effective_precision)
            effective_precision = "fp32"
        if effective_precision != "fp32":
            import torch
            if self.device.lower().startswith("cuda") and effective_precision == "bf16" and not torch.cuda.is_bf16_supported():
                log.warning("This CUDA device does not support BF16; falling back to FP32")
                effective_precision = "fp32"
            dtype = torch.float16 if effective_precision == "fp16" else torch.bfloat16
        if effective_precision != "fp32":
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        self.precision = effective_precision
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        self.model.eval()
        if qwen and query_instruction and isinstance(getattr(self.model, "prompts", None), dict):
            # Qwen's Sentence Transformers card defines a "query" prompt slot.
            self.model.prompts["query"] = f"Instruct: {query_instruction.strip()}\nQuery:"
        self.dimension = int(self.model.get_embedding_dimension())
        if self.dimension <= 0:
            raise ValueError("Embedding model reported an invalid dimension")
        self.model_revision = self._resolved_revision()

    def _resolved_revision(self) -> str | None:
        for module in self.model.modules():
            config = getattr(getattr(module, "auto_model", None), "config", None)
            revision = getattr(config, "_commit_hash", None)
            if revision:
                return str(revision)
        return None

    def _encode(self, texts: Sequence[str], method: str, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encode = getattr(self.model, method, None)
        if encode is None:
            encode = self.model.encode
        import torch
        with torch.inference_mode():
            vectors = encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (len(texts), self.dimension):
            raise ValueError(f"Embedding shape {arr.shape} does not match ({len(texts)}, {self.dimension})")
        if not np.isfinite(arr).all():
            raise ValueError("Embedding model returned non-finite values")
        return arr

    def encode_documents(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        return self._encode(texts, "encode_document", batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], "encode_query", 1)[0]
