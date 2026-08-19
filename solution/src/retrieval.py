"""Stage-1 lexical + dense candidate retrieval.

Recall is the only thing that matters here: anything this stage drops can never
be recovered by the reranker.  Two lexical views are fused because they fail
differently -- BM25 over lemmas catches inflected content words, char n-grams
catch compounds, model numbers and the morphology pymorphy3 mis-parses.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .records import Regulation
from .text import tokenize


def rrf(rank_lists: list[list[int]], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal rank fusion over ranked index lists.

    Rank-based rather than score-based on purpose: BM25 and cosine live on
    incomparable scales, and normalising them introduces a tuning knob that
    n=120 cannot support.
    """
    n = max(max(r) for r in rank_lists if r) + 1
    weights = weights or [1.0] * len(rank_lists)
    out = np.zeros(n)
    for w, ranks in zip(weights, rank_lists):
        for pos, idx in enumerate(ranks):
            out[idx] += w / (k + pos + 1)
    return out


class LexicalRetriever:
    def __init__(self, regs: list[Regulation], field: str = "coarse_text"):
        self.regs = regs
        corpus = [getattr(r, field) for r in regs]

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c) for c in corpus])

        self._char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1
        )
        self._char_mat = self._char_vec.fit_transform(corpus)

        self._word_vec = TfidfVectorizer(
            analyzer="word", tokenizer=tokenize, preprocessor=lambda x: x,
            token_pattern=None, sublinear_tf=True, min_df=1,
        )
        self._word_mat = self._word_vec.fit_transform(corpus)

    def scores(self, query: str) -> dict[str, np.ndarray]:
        bm = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        ch = (self._char_vec.transform([query]) @ self._char_mat.T).toarray().ravel()
        wd = (self._word_vec.transform([query]) @ self._word_mat.T).toarray().ravel()
        return {"bm25": bm, "char": ch, "word": wd}

    def fused(self, query: str, weights: dict[str, float] | None = None) -> np.ndarray:
        w = weights or {"bm25": 1.0, "char": 1.0, "word": 1.0}
        parts = self.scores(query)
        return rrf(
            [list(np.argsort(-parts[k])) for k in w],
            weights=[w[k] for k in w],
        )
