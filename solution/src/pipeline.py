"""The ranking pipeline, in two configurations.

Both share the same three-stage shape and differ only in capacity:

* **cpu**  lexical retrieval plus a multilingual cross-encoder over a narrow
  window.  Sized to finish inside the 30 min / 8 GB budget with no accelerator,
  because the grading host is not guaranteed to have one.
* **gpu**  the same stages with a 4B-parameter reranker at fp16 over a wider
  window.  Sized for a 16 GB card (T4 included) rather than an 8B model, whose
  ~16 GB of weights alone would exclude most graders' hardware.

Three findings from the measured sweeps are baked in here, each against my
initial expectation:

1. **Dense retrieval is not used in the CPU path.**  multilingual-e5 scored
   R@1=0.28 against the lexical stage's 0.58, and fusing it *lowered* recall at
   every weight tried.  The nomenclature is exact-terminology Russian legal
   text; general-purpose multilingual embeddings do not capture it, and the
   25-45 s they cost buys a regression.
2. **Lexical legs are not equally weighted.**  Equal-weight RRF over three legs
   scored *below its own best leg*.  Character n-grams carry this domain
   (compound terms, model numbers, morphology), so they get 3x the weight of
   the lemma leg and BM25 is dropped.
3. **The reranker is fused with stage 1, never replaces it.**  Overwriting a
   strong lexical ordering with a reranker score discards signal rather than
   refining it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .constraints import constraint_signal
from .records import Declaration, Regulation
from .retrieval import LexicalRetriever, rrf


@dataclass
class PipelineConfig:
    name: str
    # Lexical legs and their RRF weights, over the coarse (topical) view.
    lexical_weights: dict[str, float] = field(
        default_factory=lambda: {"char": 3.0, "word": 1.0}
    )
    bi_encoder: str | None = None  # off by default: measured as a regression
    dense_weight: float = 0.5
    cross_encoder: str | None = None
    ce_field: str = "fine_text"
    ce_weight: float = 1.0  # weight of the reranker leg when fused with stage 1
    n_candidates: int = 16  # stage-1 survivors reaching the reranker
    max_len_bi: int = 192
    max_len_ce: int = 256
    batch_bi: int = 32
    batch_ce: int = 16
    prefer_gpu: bool = False
    fp16: bool = False
    quantize: bool = False  # dynamic int8 on CPU: 2.5x faster, measured
    threads: int | None = None
    constraint_weight: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)


CPU_CONFIG = PipelineConfig(
    name="cpu",
    cross_encoder="BAAI/bge-reranker-v2-m3",
    ce_field="fine_text",
    ce_weight=1.0,
    n_candidates=12,
    max_len_ce=192,
    batch_ce=24,
    prefer_gpu=False,
    fp16=False,
    quantize=True,
    constraint_weight=0.02,
)

# Weights measured on a T4: the 4B reranker is strong enough here to carry
# twice the weight of the lexical leg (unlike the 568M CPU reranker at 1.0),
# and it tolerates a larger constraint bonus because its own ordering is
# already good enough that the bonus only breaks near-ties.
GPU_CONFIG = PipelineConfig(
    name="gpu",
    cross_encoder="Qwen/Qwen3-Reranker-4B",
    ce_field="fine_text",
    ce_weight=2.0,
    n_candidates=32,
    max_len_ce=512,
    batch_ce=8,  # 4B fp16 leaves ~6 GB for activations on a 15 GB T4
    prefer_gpu=True,
    fp16=True,
    constraint_weight=0.05,
)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def stage1_scores(
    cfg: PipelineConfig, regs: list[Regulation], decs: list[Declaration]
) -> np.ndarray:
    """Fused stage-1 score matrix, shape (n_decl, n_reg)."""
    t0 = time.time()
    lex = LexicalRetriever(regs, field="coarse_text")
    parts = [lex.scores(d.text) for d in decs]
    scores = np.vstack([
        rrf([list(np.argsort(-p[k])) for k in cfg.lexical_weights],
            weights=list(cfg.lexical_weights.values()))
        for p in parts
    ])
    cfg.timings["lexical"] = time.time() - t0

    if cfg.bi_encoder is None:
        return scores

    t0 = time.time()
    from .encoders import BiEncoder

    enc = BiEncoder(cfg.bi_encoder, prefer_gpu=cfg.prefer_gpu,
                    max_length=cfg.max_len_bi, batch_size=cfg.batch_bi, fp16=cfg.fp16)
    reg_emb = enc.encode([r.coarse_text for r in regs], kind="passage")
    dec_emb = enc.encode([d.text for d in decs], kind="query")
    enc.free()
    dense = dec_emb @ reg_emb.T
    cfg.timings["dense"] = time.time() - t0

    return np.vstack([
        rrf([list(np.argsort(-scores[i])), list(np.argsort(-dense[i]))],
            weights=[1.0, cfg.dense_weight])
        for i in range(len(decs))
    ])


def run_pipeline(
    cfg: PipelineConfig, regs: list[Regulation], decs: list[Declaration], *, verbose: bool = True
) -> dict[str, list[tuple[str, float]]]:
    stage1 = stage1_scores(cfg, regs, decs)
    order = np.argsort(-stage1, axis=1)
    cand_idx = [list(order[i][: cfg.n_candidates]) for i in range(len(decs))]

    ce_mat: np.ndarray | None = None
    if cfg.cross_encoder is not None:
        from .encoders import CrossEncoder

        t0 = time.time()
        if "Qwen3-Reranker" in cfg.cross_encoder:
            from .encoders import QwenReranker

            cross = QwenReranker(cfg.cross_encoder, max_length=cfg.max_len_ce,
                                 batch_size=cfg.batch_ce, fp16=cfg.fp16)
        else:
            cross = CrossEncoder(cfg.cross_encoder, prefer_gpu=cfg.prefer_gpu,
                                 max_length=cfg.max_len_ce, batch_size=cfg.batch_ce,
                                 fp16=cfg.fp16, quantize=cfg.quantize, threads=cfg.threads)
        pairs = [
            (d.text, getattr(regs[j], cfg.ce_field))
            for i, d in enumerate(decs) for j in cand_idx[i]
        ]
        ce_mat = cross.score(pairs).reshape(len(decs), cfg.n_candidates)
        cross.free()
        cfg.timings["rerank"] = time.time() - t0

    # Numeric predicates.  Both paths produce the same SATISFIED/UNKNOWN/
    # VIOLATED signal; they differ only in how the numbers are read out of the
    # text.  The LLM path extracts per *document* (360 + 120 calls), not per
    # pair, so cost does not scale with the candidate window.
    def constraint(i: int, j: int) -> float:
        return constraint_signal(decs[i].text, regs[j].disc_text)

    t0 = time.time()
    results: dict[str, list[tuple[str, float]]] = {}
    for i, d in enumerate(decs):
        cands = cand_idx[i]
        lex_leg = list(np.argsort(-stage1[i][cands]))
        if ce_mat is not None:
            fused = rrf([lex_leg, list(np.argsort(-ce_mat[i]))], weights=[1.0, cfg.ce_weight])
        else:
            fused = rrf([lex_leg], weights=[1.0])
        # Normalise by the maximum rather than min-max: RRF values sit in a
        # narrow positive band, and stretching that band to [0,1] would shrink
        # the constraint bonus below the level at which it can break a tie.
        final = fused / (fused.max() or 1.0)

        if cfg.constraint_weight:
            adj = np.array([constraint(i, j) for j in cands])
            final = final + cfg.constraint_weight * adj

        picked = sorted(range(len(cands)), key=lambda p: -final[p])[:10]
        row: list[tuple[str, float]] = []
        prev = np.inf
        for p in picked:
            # Scores must be non-increasing with rank for the submission format.
            s = min(float(final[p]), prev - 1e-9)
            row.append((regs[cands[p]].regulation_id, s))
            prev = s
        results[d.declaration_id] = row
    cfg.timings["assemble"] = time.time() - t0

    if verbose:
        total = sum(cfg.timings.values())
        parts_s = "  ".join(f"{k}={v:.1f}s" for k, v in cfg.timings.items())
        print(f"[{cfg.name}] {parts_s}  TOTAL={total:.1f}s")
    return results
