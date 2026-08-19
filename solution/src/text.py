"""Russian text normalisation for the lexical stage.

Declarations arrive in uppercase with full inflection ("МОРОЖЕНАЯ", "ЗАМОРОЖЕННОЕ")
while the nomenclature uses its own case and number, so surface-form matching
loses a large fraction of real overlap.  Lemmatising both sides recovers it.

Numbers are kept as tokens rather than stripped: "0,12", "1598", "5w-30" are
frequently the discriminating content, not noise.
"""

from __future__ import annotations

import functools
import re

_WORD = re.compile(r"[а-яёa-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)

# Legal/administrative filler that appears in nearly every nomenclature entry.
# Removing it stops BM25 from rewarding boilerplate density.
STOPWORDS = frozenset("""
и или в во на с со из для не но а также том числе прочие прочий прочая прочее
кроме если иной иные указанный указанном порядке данной данном данная этой этом
только более менее чем как так что при по от до за над под их его ее к у о об
включая включенные поименованные месте другом виде видов товарной позиции
подсубпозиции субпозиции группы группе раздела примечании дополнительном
евразийского экономического союза являются являющиеся имеющие имеющий
""".split())

_NUM = re.compile(r"(?<![\w,.])(\d+(?:[.,]\d+)?)")


@functools.lru_cache(maxsize=1)
def _analyzer():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@functools.lru_cache(maxsize=200_000)
def lemma(token: str) -> str:
    if token.isdigit() or not re.search(r"[а-яё]", token):
        return token
    return _analyzer().parse(token)[0].normal_form


def normalize(text: str) -> str:
    return text.replace("\xa0", " ").replace("ё", "е").replace("Ё", "Е").lower()


def tokenize(text: str, *, lemmatize: bool = True, drop_stopwords: bool = True) -> list[str]:
    tokens = _WORD.findall(normalize(text))
    if lemmatize:
        tokens = [lemma(t) for t in tokens]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def numbers(text: str) -> list[float]:
    """Every numeric literal in the text, comma-decimal aware."""
    out = []
    for m in _NUM.finditer(normalize(text)):
        try:
            out.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass
    return out
