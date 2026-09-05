"""Local lexical passage retrieval over a versioned, project-authored corpus."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from forge.config import PROJECT_ROOT


@dataclass(frozen=True)
class Evidence:
    passage_id: str
    title: str
    text: str
    check: str
    version: str
    corpus_sha256: str
    relevance: float
    source: str = "knowledge/playbook.json"
    authorship: str = "Project-authored analytical guidance"

    @property
    def citation(self):
        return f"{self.passage_id}@{self.version}"


class Retriever:
    def __init__(self, path: Path = PROJECT_ROOT / "knowledge/playbook.json"):
        payload = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        corpus = json.loads(payload)
        self.version = corpus["version"]
        self.passages = corpus["passages"]
        ids = [p["id"] for p in self.passages]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Evidence corpus requires unique, nonempty passage IDs.")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(
            [" ".join([p["title"], *p["sensors"], p["text"], p["check"]]) for p in self.passages]
        )

    def search(self, query: str, *, limit=3, minimum_score=0.12):
        if not 1 <= limit <= 5 or len(query) > 2000:
            raise ValueError("Retrieval request exceeds the local query budget.")
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            Evidence(
                passage_id=self.passages[i]["id"],
                title=self.passages[i]["title"],
                text=self.passages[i]["text"],
                check=self.passages[i]["check"],
                version=self.version,
                corpus_sha256=self.sha256,
                relevance=float(scores[i]),
            )
            for i in indices
            if scores[i] >= minimum_score
        ]

    def verifies(self, evidence: Evidence):
        passage = next((p for p in self.passages if p["id"] == evidence.passage_id), None)
        return bool(
            passage
            and evidence.version == self.version
            and evidence.corpus_sha256 == self.sha256
            and evidence.text == passage["text"]
            and evidence.check == passage["check"]
            and evidence.title == passage["title"]
        )
