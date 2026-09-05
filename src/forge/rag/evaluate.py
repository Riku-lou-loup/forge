"""Evaluate the small disclosed retrieval regression set without any API calls."""

import json

from forge.config import PROJECT_ROOT
from forge.rag.retrieval import Retriever


def evaluate_retrieval(root=PROJECT_ROOT):
    suite = json.loads((root / "knowledge/retrieval-cases.json").read_text())
    retriever = Retriever(root / "knowledge/playbook.json")
    results = []
    recalls, abstentions = [], []
    for case in suite["cases"]:
        retrieved = retriever.search(case["query"], limit=3)
        actual = [e.passage_id for e in retrieved]
        expected = set(case["expected"])
        recall = len(expected & set(actual)) / len(expected) if expected else None
        if expected:
            recalls.append(recall)
        else:
            abstentions.append(not actual)
        results.append(
            {
                **case,
                "retrieved": actual,
                "recall_at_3": recall,
                "citations_verified": all(retriever.verifies(e) for e in retrieved),
            }
        )
    return {
        "purpose": suite["purpose"],
        "corpus_sha256": retriever.sha256,
        "recall_at_3": sum(recalls) / len(recalls),
        "expected_abstentions": len(abstentions),
        "correct_abstentions": sum(abstentions),
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_retrieval(), indent=2))
