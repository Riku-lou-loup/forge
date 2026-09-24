"""Evaluate the small disclosed retrieval regression set without any API calls."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from forge.config import PROJECT_ROOT
from forge.rag.retrieval import Retriever


def _configuration(retriever, minimum_score):
    return {
        "backend": retriever.backend,
        "minimum_score": retriever.minimum_score if minimum_score is None else minimum_score,
        "parameters": (
            {"k1": retriever.bm25.k1, "b": retriever.bm25.b} if retriever.backend == "bm25" else {}
        ),
        "corpus_sha256": retriever.sha256,
    }


def evaluate_retrieval(root=PROJECT_ROOT, *, backend="tfidf", minimum_score=None, k1=1.2, b=0.75):
    root = Path(root)
    suite = json.loads((root / "knowledge/retrieval-cases.json").read_text(encoding="utf-8"))
    retriever = Retriever(root / "knowledge/playbook.json", backend=backend, k1=k1, b=b)
    results = []
    recalls, reciprocal_ranks, abstentions = [], [], []
    verified_citations = retrieved_citations = 0
    for case in suite["cases"]:
        retrieved = retriever.search(case["query"], limit=3, minimum_score=minimum_score)
        actual = [e.passage_id for e in retrieved]
        expected = set(case["expected"])
        recall = len(expected & set(actual)) / len(expected) if expected else None
        reciprocal_rank = (
            next((1 / rank for rank, pid in enumerate(actual, 1) if pid in expected), 0.0)
            if expected
            else None
        )
        if expected:
            recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank)
        else:
            abstentions.append(not actual)
        verified = [retriever.verifies(e) for e in retrieved]
        verified_citations += sum(verified)
        retrieved_citations += len(retrieved)
        results.append(
            {
                **case,
                "retrieved": actual,
                "scores": [e.relevance for e in retrieved],
                "recall_at_3": recall,
                "reciprocal_rank": reciprocal_rank,
                "correct_abstention": not actual if not expected else None,
                "citations_verified": all(verified),
            }
        )
    return {
        "purpose": suite["purpose"],
        **_configuration(retriever, minimum_score),
        "recall_at_3": sum(recalls) / len(recalls) if recalls else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "ranking_metric_scope": "Answerable cases only, using the first three thresholded results.",
        "expected_abstentions": len(abstentions),
        "correct_abstentions": sum(abstentions),
        "retrieved_citations": retrieved_citations,
        "verified_citations": verified_citations,
        "citations_verified": verified_citations == retrieved_citations,
        "results": results,
    }


def query_retrieval(
    query, root=PROJECT_ROOT, *, backend="tfidf", minimum_score=None, k1=1.2, b=0.75
):
    retriever = Retriever(Path(root) / "knowledge/playbook.json", backend=backend, k1=k1, b=b)
    evidence = retriever.search(query, minimum_score=minimum_score)
    return {
        **_configuration(retriever, minimum_score),
        "query": query,
        "abstained": not evidence,
        "results": [
            {**asdict(item), "citation": item.citation, "verified": retriever.verifies(item)}
            for item in evidence
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="FORGE repository root")
    parser.add_argument("--backend", choices=("tfidf", "bm25"), default="tfidf")
    parser.add_argument(
        "--compare", action="store_true", help="Run both backends with separate defaults"
    )
    parser.add_argument("--query", help="Search this query instead of the nine regression cases")
    parser.add_argument(
        "--minimum-score", type=float, help="Override the selected backend's threshold"
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 term-frequency saturation")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 document-length normalization")
    args = parser.parse_args(argv)
    if args.compare and args.minimum_score is not None:
        parser.error("Use --minimum-score with a single backend; scores have different scales.")
    operation = evaluate_retrieval if args.query is None else query_retrieval
    parameters = {
        "root": args.root,
        "minimum_score": args.minimum_score,
        "k1": args.k1,
        "b": args.b,
    }
    if args.query is not None:
        parameters["query"] = args.query
    try:
        if args.compare:
            result = {
                "backends": {
                    backend: operation(backend=backend, **parameters)
                    for backend in ("tfidf", "bm25")
                }
            }
        else:
            result = operation(backend=args.backend, **parameters)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
