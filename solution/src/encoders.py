"""Bi-encoder and cross-encoder wrappers, loaded strictly from the local cache.

Everything resolves through `models/lockfile.json`, and `HF_HUB_OFFLINE` is set
before any transformers import, so a scored run cannot reach the network even
if a cache entry is missing -- it fails loudly instead of silently downloading.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"
_LOCKFILE = _ROOT / "model_lock.json"

# Must be set before transformers/huggingface_hub are imported anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _lock() -> dict[str, str]:
    if not _LOCKFILE.exists():
        raise FileNotFoundError(
            f"{_LOCKFILE} not found — run `python prepare_models.py` once, with network."
        )
    return json.loads(_LOCKFILE.read_text(encoding="utf-8"))


def resolve(repo: str) -> str:
    """Local snapshot directory for a repo, pinned to its locked commit."""
    sha = _lock().get(repo)
    if sha is None:
        raise KeyError(f"{repo} is not in the lockfile — re-run prepare_models.py")
    path = _MODELS_DIR / f"models--{repo.replace('/', '--')}" / "snapshots" / sha
    if not path.is_dir():
        raise FileNotFoundError(f"weights for {repo}@{sha} missing at {path}")
    return str(path)


def _device(prefer_gpu: bool) -> str:
    import torch

    return "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"


class BiEncoder:
    """Dense retrieval encoder.

    E5 checkpoints are asymmetric and expect "query: " / "passage: " prefixes;
    omitting them measurably degrades retrieval, so the prefix is applied here
    rather than left to the caller.
    """

    def __init__(self, repo: str, *, prefer_gpu: bool = False, max_length: int = 192,
                 batch_size: int = 32, fp16: bool = False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.repo = repo
        self.device = _device(prefer_gpu)
        self.max_length = max_length
        self.batch_size = batch_size
        self._is_e5 = "e5" in repo.lower()

        path = resolve(repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if not self._is_e5:
            return texts
        tag = "query: " if kind == "query" else "passage: "
        return [tag + t for t in texts]

    @staticmethod
    def _mean_pool(hidden, mask):
        import torch

        m = mask.unsqueeze(-1).to(hidden.dtype)
        return torch.sum(hidden * m, 1) / torch.clamp(m.sum(1), min=1e-9)

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        import torch

        prepared = self._prefix(texts, kind)
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(prepared), self.batch_size):
                batch = prepared[i:i + self.batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                vec = self._mean_pool(hidden, enc["attention_mask"])
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                out.append(vec.float().cpu().numpy())
        emb = np.vstack(out)
        if not np.isfinite(emb).all():
            raise FloatingPointError(f"{self.repo}: non-finite embeddings (fp16 overflow?)")
        return emb

    def free(self) -> None:
        """Release weights: the 8 GB RAM cap does not allow co-resident models."""
        import gc

        import torch

        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


class CrossEncoder:
    """Pairwise reranker producing one relevance logit per (query, doc) pair."""

    def __init__(self, repo: str, *, prefer_gpu: bool = False, max_length: int = 256,
                 batch_size: int = 16, fp16: bool = False, quantize: bool = False,
                 threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.repo = repo
        self.device = _device(prefer_gpu)
        self.max_length = max_length
        self.batch_size = batch_size

        if self.device == "cpu" and threads:
            torch.set_num_threads(threads)

        path = resolve(repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(path, dtype=dtype)
            .to(self.device).eval()
        )
        if quantize and self.device == "cpu":
            # Dynamic int8 over Linear layers: the matmuls dominate encoder
            # inference, and weight-only quantisation needs no calibration set,
            # so it stays reproducible.  Measured 1.8 -> 4.5 pairs/s.
            import gc

            fp32 = self.model
            self.model = torch.quantization.quantize_dynamic(
                fp32, {torch.nn.Linear}, dtype=torch.qint8
            )
            # Both copies are resident during conversion; drop the fp32 one
            # promptly because the 8 GB cap is a hard failure, not a target.
            del fp32
            gc.collect()

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        out: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                enc = self.tok([p[0] for p in batch], [p[1] for p in batch],
                               padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(logits[:, 0].float().cpu().numpy())
        scores = np.concatenate(out) if out else np.zeros(0)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.repo}: non-finite scores (fp16 overflow?)")
        return scores

    def free(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


class QwenReranker:
    """Qwen3-Reranker: a causal LM scored on its yes/no logit, not a classifier.

    The checkpoint is trained to answer a single token after a fixed chat
    template, so relevance is read off the logit gap between "yes" and "no" at
    the final position rather than from a classification head.  Getting the
    template wrong does not error -- it silently produces near-random scores --
    so the prefix/suffix below are reproduced exactly as published.
    """

    INSTRUCTION = (
        "Given a customs goods declaration, judge whether the candidate TN VED "
        "tariff code is the correct classification for it. Pay attention to "
        "numeric thresholds, negations, condition (new/used) and material "
        "composition."
    )
    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, repo: str, *, max_length: int = 512, batch_size: int = 8,
                 fp16: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.repo = repo
        self.device = _device(True)
        if self.device != "cuda":
            raise RuntimeError("QwenReranker is a GPU-only path; use CrossEncoder on CPU")
        self.max_length = max_length
        self.batch_size = batch_size

        path = resolve(repo)
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        # T4 is Turing: no native bf16 tensor cores, so fp16 is the fast path.
        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.device).eval()

        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self._pre_ids = self.tok.encode(self._PREFIX, add_special_tokens=False)
        self._suf_ids = self.tok.encode(self._SUFFIX, add_special_tokens=False)

    def _build(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        budget = self.max_length - len(self._pre_ids) - len(self._suf_ids)
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                texts = [self._build(q, d) for q, d in chunk]
                # return_attention_mask=False is load-bearing: the template
                # tokens are spliced in *after* tokenisation, so a mask built
                # from the pre-splice lengths would mark real prefix/suffix
                # tokens as padding.  Letting pad() regenerate it from the
                # final ids is the only consistent option -- and getting this
                # wrong degrades silently to random ranking rather than erroring.
                enc = self.tok(texts, truncation=True, max_length=budget,
                               add_special_tokens=False, return_attention_mask=False)
                enc["input_ids"] = [
                    self._pre_ids + ids + self._suf_ids for ids in enc["input_ids"]
                ]
                enc = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits[:, -1, :]
                pair_logits = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
                probs = torch.nn.functional.log_softmax(pair_logits.float(), dim=1)[:, 1]
                out.append(probs.exp().cpu().numpy())
        scores = np.concatenate(out) if out else np.zeros(0)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.repo}: non-finite scores (fp16 overflow?)")
        return scores

    def free(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
