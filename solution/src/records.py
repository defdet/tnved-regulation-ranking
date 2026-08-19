"""Canonical regulation / declaration records.

Two views are built per regulation because the two retrieval stages need
different things from the same document:

* **coarse** — broad topical text (section, chapter, heading, explanation,
  full description).  Optimised for recall: get the right product family.
* **fine** — context plus discriminators (inherited qualifier chain, own leaf
  label, gloss, and the description with its shared sibling prefix removed).
  Optimised for precision: separate near-identical siblings.  Shared family
  context is retained on purpose — a cross-encoder attends across the pair and
  uses it to locate the comparison.
* **disc** — the residue that is *unique* within the sibling block: qualifiers
  not shared by every sibling, own leaf label, and the divergent description
  tail.  A bi-encoder collapses a document to one vector, so shared mass
  dominates it; this view strips that mass out.  It also feeds the numeric
  constraint parser, which wants exactly the predicates that discriminate.

The split exists because hard negatives here share a very long boilerplate
prefix — raw `description` cosine between two siblings runs ~0.99, so the
discriminating tail is drowned unless it is isolated.

`notes` is deliberately unused: only 93 distinct values across 360 rows, it is
identical within every sibling block and hard-truncated at 1000 chars, so it
carries no discriminative signal and dilutes whatever it is concatenated to.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .hierarchy import ancestor_indices, parse_dump


@dataclass
class Regulation:
    regulation_id: str
    code: str
    chapter: str  # 2-digit
    heading: str  # 4-digit
    subheading: str  # 6-digit
    chapter_label: str
    heading_label: str
    path: list[str]  # inherited qualifier chain, heading-exclusive
    leaf_label: str
    gloss: str  # the "[...]" normalized name from the dump
    description: str
    desc_tail: str  # description minus prefix shared with pool siblings
    explanation: str
    coarse_text: str
    fine_text: str
    disc_text: str


def _common_prefix_len(strings: list[str]) -> int:
    if len(strings) < 2:
        return 0
    first, last = min(strings), max(strings)
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return i


def _trim_to_word(text: str, start: int) -> str:
    """Cut at `start`, backing off to the *preceding* word boundary.

    Backwards, not forwards: the character at `start` is usually mid-word, and
    that word is where the siblings first diverge — advancing would delete the
    very token that discriminates them.
    """
    if start <= 0 or start >= len(text):
        return text
    while start > 0 and text[start - 1] not in " ,;:":
        start -= 1
    return text[start:].lstrip(" ,;:").strip()


def _short_family(label: str, limit: int = 110) -> str:
    """A compact product-family hint for the fine view.

    Heading labels and some subheading labels run to 300+ chars of legal
    boilerplate ("...кроме сырых; продукты, в другом месте не поименованные..."),
    which is exactly the dilution the fine view exists to avoid.  Keep the
    leading clause only.  Genuinely discriminative qualifiers are short and
    specific ("более 1500 см3, но не более 3000 см3") and survive untouched.
    """
    head = re.split(r"[;(]", label, maxsplit=1)[0].strip()
    if len(head) > limit:
        head = head[:limit].rsplit(" ", 1)[0] + "…"
    return head


def load_regulations(data_dir: str) -> list[Regulation]:
    nodes, by_code = parse_dump(os.path.join(data_dir, "tnved_knowledge.txt"))
    with open(os.path.join(data_dir, "regulations.jsonl"), encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]

    # Shared-prefix removal is computed within the 4-digit heading block, which
    # is the level at which this pool's hard negatives are drawn.
    by_heading: dict[str, list[str]] = {}
    for r in raw:
        by_heading.setdefault(r["code"][:4], []).append(r["description"])
    prefix_len = {h: _common_prefix_len(d) for h, d in by_heading.items()}

    records_raw: list[Regulation] = []
    for r in raw:
        code = r["code"]
        idx = by_code.get(code)
        # Qualifiers are the ancestors strictly below the 4-digit heading
        # (depth 2).  Selecting by depth rather than slicing a fixed count
        # keeps chains intact when a level is missing from the dump.
        qualifiers = (
            [nodes[a].label for a in ancestor_indices(nodes, idx)
             if nodes[a].depth > 2 and nodes[a].label]
            if idx is not None else []
        )
        leaf_label = nodes[idx].label if idx is not None else ""
        gloss = (nodes[idx].normalized if idx is not None else None) or ""

        ch_idx, hd_idx = by_code.get(code[:2]), by_code.get(code[:4])
        chapter_label = nodes[ch_idx].label if ch_idx is not None else ""
        heading_label = nodes[hd_idx].label if hd_idx is not None else ""

        desc = (r.get("description") or "").strip()
        tail = _trim_to_word(desc, prefix_len.get(code[:4], 0))

        coarse = " | ".join(
            p for p in [chapter_label, heading_label, (r.get("explanation") or "").strip(), desc] if p
        )
        fine = " | ".join(
            p
            for p in [
                _short_family(heading_label),
                " › ".join(_short_family(q, 150) for q in qualifiers),
                leaf_label,
                gloss if gloss.lower() != desc.lower() else "",
                tail if tail and tail != desc else "",
            ]
            if p
        )

        records_raw.append(
            Regulation(
                regulation_id=r["regulation_id"],
                code=code,
                chapter=code[:2],
                heading=code[:4],
                subheading=code[:6],
                chapter_label=chapter_label,
                heading_label=heading_label,
                path=qualifiers,
                leaf_label=leaf_label,
                gloss=gloss,
                description=desc,
                desc_tail=tail,
                explanation=(r.get("explanation") or "").strip(),
                coarse_text=coarse,
                fine_text=fine,
                disc_text="",
            )
        )

    # Second pass: a qualifier only discriminates if some sibling lacks it.
    blocks: dict[str, list[Regulation]] = {}
    for rec in records_raw:
        blocks.setdefault(rec.heading, []).append(rec)
    for block in blocks.values():
        shared = set.intersection(*(set(r.path) for r in block)) if len(block) > 1 else set()
        for rec in block:
            own = [q for q in rec.path if q not in shared]
            rec.disc_text = " | ".join(
                p for p in [
                    " › ".join(_short_family(q, 150) for q in own),
                    rec.leaf_label,
                    rec.desc_tail if rec.desc_tail != rec.description else "",
                ] if p
            ) or rec.leaf_label or rec.desc_tail or rec.description
    return records_raw


@dataclass
class Declaration:
    declaration_id: str
    text: str  # G31_1 + desc_extention, the only fields carrying signal
    g31: str
    ext: str
    direction: str
    country: str


def load_declarations(data_dir: str) -> list[Declaration]:
    with open(os.path.join(data_dir, "declarations.jsonl"), encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    out = []
    for d in raw:
        g31 = (d.get("G31_1") or "").strip()
        ext = (d.get("desc_extention") or "").strip()
        out.append(
            Declaration(
                declaration_id=d["declaration_id"],
                text=f"{g31}. {ext}" if ext else g31,
                g31=g31,
                ext=ext,
                direction=(d.get("G011") or "").strip(),
                country=(d.get("G34") or "").strip(),
            )
        )
    return out


if __name__ == "__main__":
    regs = load_regulations("data")
    decs = load_declarations("data")
    print(f"{len(regs)} regulations, {len(decs)} declarations")
    empty_path = sum(1 for r in regs if not r.path)
    print(f"no qualifier chain: {empty_path}  |  no gloss: {sum(1 for r in regs if not r.gloss)}")
    import statistics as st
    print("fine_text len  med:", int(st.median([len(r.fine_text) for r in regs])),
          " max:", max(len(r.fine_text) for r in regs))
    for c in ("2710194600", "8703606021", "0203295909"):
        r = next(x for x in regs if x.code == c)
        print(f"\n--- {r.regulation_id} {c}\nFINE: {r.fine_text[:400]}")
