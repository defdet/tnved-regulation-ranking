"""Hard format gate for out/predictions.csv.

The assignment scores a malformed submission as a failure regardless of ranking
quality -- missing rows, unknown ids, repeated regulations and repeated ranks
are all format errors -- so this runs on every execution, not just in tests.
"""

from __future__ import annotations

import csv
import math

REQUIRED_COLUMNS = ["declaration_id", "rank", "regulation_id", "score"]


def validate_predictions(path: str, declaration_ids: set[str], regulation_ids: set[str]) -> None:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError("predictions.csv is empty")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    by_decl: dict[str, list[dict]] = {}
    for r in rows:
        by_decl.setdefault(r["declaration_id"], []).append(r)

    if unknown := set(by_decl) - declaration_ids:
        raise ValueError(f"unknown declaration_id(s): {sorted(unknown)[:5]}")
    if absent := declaration_ids - set(by_decl):
        raise ValueError(f"{len(absent)} declaration(s) with no rows, e.g. {sorted(absent)[:5]}")

    for did, group in by_decl.items():
        if len(group) != 10:
            raise ValueError(f"{did}: expected 10 rows, got {len(group)}")
        if sorted(int(r["rank"]) for r in group) != list(range(1, 11)):
            raise ValueError(f"{did}: ranks must be exactly 1..10 with no repeats")
        regs = [r["regulation_id"] for r in group]
        if len(set(regs)) != 10:
            raise ValueError(f"{did}: repeated regulation_id in top-10")
        if bad := set(regs) - regulation_ids:
            raise ValueError(f"{did}: unknown regulation_id(s) {sorted(bad)[:3]}")
        for r in group:
            v = float(r["score"])
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{did}: non-finite score {r['score']!r}")
        ordered = sorted(group, key=lambda r: int(r["rank"]))
        scores = [float(r["score"]) for r in ordered]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"{did}: score must be non-increasing as rank increases")
