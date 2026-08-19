"""Evaluation metrics for the ranking task.

The hidden ground truth names exactly one relevant regulation per declaration,
so the whole metric family reduces to "where did the single gold item land".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Scores:
    n: int
    hit1: float
    hit3: float
    hit5: float
    hit10: float
    mrr10: float

    def __str__(self) -> str:
        return (
            f"n={self.n}  Hit@1={self.hit1:.3f}  Hit@3={self.hit3:.3f}  "
            f"Hit@5={self.hit5:.3f}  Hit@10={self.hit10:.3f}  MRR@10={self.mrr10:.3f}"
        )


def evaluate(ranking: dict[str, list[str]], gold: dict[str, str]) -> Scores:
    """`ranking` maps declaration_id -> ordered regulation_ids (best first)."""
    ranks: list[int | None] = []
    for did, g in gold.items():
        order = ranking.get(did, [])
        ranks.append(order.index(g) + 1 if g in order else None)

    def hit(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = sum(1.0 / r for r in ranks if r is not None and r <= 10) / len(ranks)
    return Scores(len(ranks), hit(1), hit(3), hit(5), hit(10), mrr)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the honest error bar on a Hit@k at small n.

    Reported because n=120 puts roughly +/-0.09 around a mid-range Hit@1; a
    difference smaller than that is not evidence of anything.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def report(ranking: dict[str, list[str]], gold: dict[str, str], label: str = "") -> Scores:
    s = evaluate(ranking, gold)
    lo, hi = wilson_interval(round(s.hit1 * s.n), s.n)
    print(f"{label:28s} {s}   [Hit@1 95% CI {lo:.2f}-{hi:.2f}]")
    return s
