"""Numeric interval predicates parsed out of nomenclature text.

A large share of this pool's hard negatives differ by exactly one threshold::

    2710194290  ... с содержанием серы не более 0,05 мас.%
    2710194600  ... серы более 0,05 мас.%, но не более 0,2 мас.%   <- 0,12 lands here
    2710194800  ... серы более 0,2 мас.%

Embedding cosine cannot order 0,05 / 0,12 / 0,2, and a cross-encoder does it
only unreliably.  Parsing the interval and evaluating it is exact where it
fires.

Design rule: this is a *feature*, never a filter.  A failed parse, an unmatched
unit or an unmentioned quantity all resolve to UNKNOWN and cost the candidate
nothing.  Anything stricter risks eliminating the gold on a regex miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import STOPWORDS, lemma, normalize

SATISFIED, UNKNOWN, VIOLATED = 1.0, 0.0, -1.0

# Surface unit -> canonical unit.  Longer keys first, so "мас.%" is not
# shadowed by a bare "%".
_UNITS: list[tuple[str, str]] = [
    ("мас.%", "pct_mass"), ("мас. %", "pct_mass"), ("масс.%", "pct_mass"),
    ("об.%", "pct_vol"), ("об. %", "pct_vol"),
    ("см3", "cm3"), ("куб.см", "cm3"),
    ("г/м2", "gsm"),
    ("дтекс", "dtex"), ("текс", "tex"),
    ("мкм", "um"), ("мм", "mm"), ("квт", "kw"), ("мпа", "mpa"), ("кг", "kg"),
    ("лет", "year"), ("года", "year"), ("год", "year"),
    ("дюйм", "inch"), ("карат", "carat"), ("%", "pct"),
]

# comparator -> (is_lower_bound, is_strict)
_CMP: dict[str, tuple[bool, bool]] = {
    "более": (True, True), "свыше": (True, True), "выше": (True, True),
    "не менее": (True, False), "от": (True, False),
    "менее": (False, True), "ниже": (False, True),
    "не более": (False, False), "до": (False, False), "не превышает": (False, False),
}

_CMP_RE = re.compile(
    r"(не более|не менее|не превышает|более|менее|свыше|выше|ниже|от|до)\s*"
    r"(\d+(?:[.,]\d+)?)\s*([а-яё0-9./%]*)"
)
_QTY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([а-яё0-9./%]+)")


def _canon_unit(raw: str) -> str | None:
    raw = raw.strip(" .,;")
    for surface, canon in _UNITS:
        if raw.startswith(surface):
            return canon
    return None


def _subject(text: str, start: int, width: int = 60) -> frozenset[str]:
    """Content lemmas immediately preceding a number: what it is *about*.

    Units alone are ambiguous.  A car declaration carries both an engine
    displacement and a mileage; a board carries thickness, width and length.
    The surrounding words are what tie a quantity to the right predicate.
    """
    window = text[max(0, start - width):start]
    toks = re.findall(r"[а-яё]+", window)
    return frozenset(lemma(t) for t in toks if t not in STOPWORDS and len(t) > 2)


@dataclass
class Interval:
    unit: str
    lo: float | None = None
    lo_strict: bool = True
    hi: float | None = None
    hi_strict: bool = False
    subject: frozenset[str] = field(default_factory=frozenset)

    def contains(self, value: float) -> bool:
        if self.lo is not None:
            if (value <= self.lo) if self.lo_strict else (value < self.lo):
                return False
        if self.hi is not None:
            if (value >= self.hi) if self.hi_strict else (value > self.hi):
                return False
        return True


@dataclass
class Quantity:
    unit: str
    value: float
    subject: frozenset[str] = field(default_factory=frozenset)


def parse_intervals(text: str) -> list[Interval]:
    """Intervals stated in nomenclature text.

    A lower bound followed by an upper bound of the same unit is merged into
    one closed interval, which is what turns the very common
    "более X мас.%, но не более Y мас.%" into a single range rather than two
    contradictory open ones.  The reverse order is deliberately *not* merged --
    see the comment below.
    """
    t = normalize(text)
    found: list[Interval] = []
    for m in _CMP_RE.finditer(t):
        unit = _canon_unit(m.group(3))
        if unit is None:
            continue
        try:
            value = float(m.group(2).replace(",", "."))
        except ValueError:
            continue
        is_lower, strict = _CMP[m.group(1)]
        subj = _subject(t, m.start())
        # Merge only the canonical "более X, но не более Y" order: a lower
        # bound already open, closed by the next upper bound.  The reverse
        # ("не более 40 ... более 47") is not a range at all -- it is two
        # predicates about *different* quantities (fat, then moisture), and
        # merging them yields the impossible interval (47, 40].
        if found and found[-1].unit == unit and not is_lower:
            prev = found[-1]
            if prev.hi is None and prev.lo is not None:
                prev.hi, prev.hi_strict = value, strict
                prev.subject = prev.subject | subj
                continue
        found.append(
            Interval(unit, lo=value, lo_strict=strict, subject=subj)
            if is_lower
            else Interval(unit, hi=value, hi_strict=strict, subject=subj)
        )
    return found


def parse_quantities(text: str) -> list[Quantity]:
    t = normalize(text)
    out: list[Quantity] = []
    for m in _QTY_RE.finditer(t):
        unit = _canon_unit(m.group(2))
        if unit is None:
            continue
        try:
            value = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        out.append(Quantity(unit, value, _subject(t, m.start())))
    return out


_PCT_UNITS = ("pct_mass", "pct_vol", "pct")


def constraint_signal(decl_text: str, cand_text: str) -> float:
    """SATISFIED / UNKNOWN / VIOLATED for one (declaration, candidate) pair.

    When several declared quantities share a unit, the one whose surrounding
    words overlap the predicate's subject wins.  A tie abstains rather than
    guessing: a wrong VIOLATED is far more expensive than an UNKNOWN.
    """
    intervals = parse_intervals(cand_text)
    if not intervals:
        return UNKNOWN
    quantities = parse_quantities(decl_text)
    if not quantities:
        return UNKNOWN

    verdicts: list[float] = []
    for iv in intervals:
        same = [q for q in quantities if q.unit == iv.unit]
        if not same and iv.unit in _PCT_UNITS:
            same = [q for q in quantities if q.unit in _PCT_UNITS]
        if not same:
            continue
        if len(same) > 1:
            ranked = sorted(same, key=lambda q: len(q.subject & iv.subject), reverse=True)
            best = len(ranked[0].subject & iv.subject)
            if best == 0 or len(ranked[1].subject & iv.subject) == best:
                continue  # genuinely ambiguous -> abstain
            chosen = ranked[0]
        else:
            chosen = same[0]
        verdicts.append(SATISFIED if iv.contains(chosen.value) else VIOLATED)

    if not verdicts:
        return UNKNOWN
    if any(v == VIOLATED for v in verdicts):
        return VIOLATED
    return SATISFIED
