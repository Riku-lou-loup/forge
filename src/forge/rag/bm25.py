"""Positive-IDF Okapi BM25 over exact, normalized unigram tokens."""

import math
from collections import Counter
from numbers import Real

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    """A small in-memory index. Repeated query tokens count once."""

    def __init__(self, documents, *, k1=1.2, b=0.75):
        if isinstance(k1, bool) or not isinstance(k1, Real) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError("BM25 k1 must be finite and greater than zero.")
        if (
            isinstance(b, bool)
            or not isinstance(b, Real)
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise ValueError("BM25 b must be finite and between zero and one.")
        self.k1, self.b = float(k1), float(b)
        # Match the TF-IDF backend's lowercase/token/stop-word rules, using unigrams.
        self.analyze = CountVectorizer(stop_words="english").build_analyzer()
        self.frequencies = [Counter(self.analyze(document)) for document in documents]
        lengths = np.array([sum(counts.values()) for counts in self.frequencies], dtype=float)
        self.average_length = float(lengths.mean()) if len(lengths) else 0.0
        relative_lengths = lengths / self.average_length if self.average_length else lengths
        self.normalization = 1 - self.b + self.b * relative_lengths
        document_frequency = Counter(term for counts in self.frequencies for term in counts)
        n = len(self.frequencies)
        self.idf = {
            term: math.log1p((n - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def score(self, query):
        scores = np.zeros(len(self.frequencies), dtype=float)
        # Divide numerator and denominator by the same scale to avoid overflow
        # at very large k1 without changing the BM25 formula.
        scale = max(1.0, self.k1)
        scaled_k1 = self.k1 / scale
        # Sorted terms make floating-point summation independent of hash randomization.
        for term in sorted(set(self.analyze(query)) & self.idf.keys()):
            frequencies = np.array([counts.get(term, 0) for counts in self.frequencies])
            matched = frequencies > 0
            scores[matched] += self.idf[term] * (
                frequencies[matched]
                * (scaled_k1 + 1 / scale)
                / (frequencies[matched] / scale + scaled_k1 * self.normalization[matched])
            )
        return scores
