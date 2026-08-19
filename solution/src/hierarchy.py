"""Parser for the TN VED EAEU reference dump (`tnved_knowledge.txt`).

The dump is a depth-encoded tree.  Every meaningful line has the shape

    <indent><code?> | <dash-run><label>[ [normalized gloss]]

Depth is carried by the **leading dash run**, not by the indentation.  The two
disagree often enough to matter: `8703606010` sits at indent 12 with 4 dashes
while its sibling `– – – – прочие:` sits at indent 13 with the same 4 dashes, so
an indent-driven parser adopts the sibling as a parent and corrupts the chain.
Lines with no dash run are disambiguated by code width instead — section
(roman), chapter (2 digits), heading (4 digits).

Recovering the chain is the whole point of this module.  A 10-digit code's own
label is very often just "прочие" ("other"); every qualifier that discriminates
it from its siblings is inherited.  Flattening the chain onto the leaf is what
makes sibling codes separable at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NODE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<code>\d{2,10}|[IVXLC]+)?[ \t]*\|[ \t]?(?P<rest>.*)$"
)
_DASH_RUN = re.compile(r"^(?:[–\-—][ \t]*)+")
_BRACKET = re.compile(r"\[([^\[\]]*)\]\s*$")
_SECTION_HIER = "ИЕРАРХИЯ ТН ВЭД"
_SECTION_END = ("ПРИМЕЧАНИЯ", "ПОЯСНЕНИЯ")

# Depth assigned to dash-less lines, by code width.  Dashed lines sit at
# HEADING_DEPTH + <number of dashes>, which keeps one global ordering.
_SECTION_DEPTH, _CHAPTER_DEPTH, _HEADING_DEPTH = 0, 1, 2


@dataclass
class Node:
    code: str | None
    label: str  # own label, dash run stripped
    normalized: str | None  # the "[...]" gloss, when the dump supplies one
    depth: int
    parent: int | None
    children: list[int] = field(default_factory=list)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_dump(path: str) -> tuple[list[Node], dict[str, int]]:
    """Return (nodes, code -> node index).

    Only the ИЕРАРХИЯ blocks are walked.  The ПРИМЕЧАНИЯ / ПОЯСНЕНИЯ prose
    blocks carry no per-code structure and would inject noise into the chains.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    nodes: list[Node] = []
    by_code: dict[str, int] = {}
    stack: list[int] = []  # current ancestor path
    in_hier = False

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(_SECTION_HIER):
            in_hier = True
            stack.clear()
            continue
        if stripped.startswith(_SECTION_END):
            in_hier = False
            continue
        if not in_hier or "|" not in raw:
            continue

        m = _NODE.match(raw.replace("\xa0", " "))
        if not m:
            continue

        code = m.group("code")
        rest = m.group("rest")

        normalized = None
        if (bm := _BRACKET.search(rest)) is not None:
            normalized = _norm_ws(bm.group(1))
            rest = rest[: bm.start()]

        dash_m = _DASH_RUN.match(rest.replace("\xa0", " ").lstrip())
        n_dashes = len(re.findall(r"[–\-—]", dash_m.group(0))) if dash_m else 0
        label = _norm_ws(_DASH_RUN.sub("", rest.replace("\xa0", " ").lstrip())).rstrip(":").strip()

        if n_dashes:
            depth = _HEADING_DEPTH + n_dashes
        elif code is None or not code.isdigit():
            depth = _SECTION_DEPTH
        elif len(code) <= 2:
            depth = _CHAPTER_DEPTH
        else:
            depth = _HEADING_DEPTH

        if not label and code is None:
            continue

        while stack and nodes[stack[-1]].depth >= depth:
            stack.pop()

        idx = len(nodes)
        nodes.append(
            Node(
                code=code if (code and code.isdigit()) else None,
                label=label,
                normalized=normalized,
                depth=depth,
                parent=stack[-1] if stack else None,
            )
        )
        if stack:
            nodes[stack[-1]].children.append(idx)
        stack.append(idx)

        # First occurrence wins: the dump reprints a few codes across sections,
        # and the first is the one sitting inside its own subtree.
        if code and code.isdigit() and code not in by_code:
            by_code[code] = idx

    return nodes, by_code


def ancestor_labels(nodes: list[Node], idx: int, include_self: bool = False) -> list[str]:
    """Labels from outermost ancestor down to (optionally) the node itself."""
    chain: list[str] = []
    cur: int | None = idx if include_self else nodes[idx].parent
    while cur is not None:
        if nodes[cur].label:
            chain.append(nodes[cur].label)
        cur = nodes[cur].parent
    return list(reversed(chain))


def ancestor_indices(nodes: list[Node], idx: int, include_self: bool = False) -> list[int]:
    """Ancestor node indices, outermost first."""
    chain: list[int] = []
    cur: int | None = idx if include_self else nodes[idx].parent
    while cur is not None:
        chain.append(cur)
        cur = nodes[cur].parent
    return list(reversed(chain))
