"""BM25 ranking, independent thresholds, and shared evidence safeguards."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from forge.agents.workflow import investigate
from forge.data.datasets import FEATURES
from forge.ml.detectors import Detector
from forge.rag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]


def corpus(tmp_path, texts):
    path = tmp_path / "playbook.json"
    path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "passages": [
                    {"id": f"p{i}", "title": "", "sensors": [], "text": text, "check": ""}
                    for i, text in enumerate(texts)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_bm25_matches_hand_calculated_positive_idf(tmp_path):
    path = corpus(tmp_path, ["pump pump pressure", "pressure"])
    retriever = Retriever(path, backend="bm25", k1=1.2, b=0.75)
    found = retriever.search("pump", minimum_score=0)
    assert [e.passage_id for e in found] == ["p0"]
    # N=2, df(pump)=1, tf=2, length=3, average length=2.
    expected = math.log(2) * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert found[0].relevance == pytest.approx(expected)


def test_document_frequency_remains_positive_for_universal_term(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["pump", "pump"]), backend="bm25")
    found = retriever.search("pump", minimum_score=0)
    assert len(found) == 2
    assert [e.relevance for e in found] == pytest.approx([math.log(1.2)] * 2)


def test_document_length_penalty_can_be_disabled(tmp_path):
    path = corpus(tmp_path, ["pump " + "motor " * 30, "pump"])
    normalized = Retriever(path, backend="bm25", b=0.75)
    unnormalized = Retriever(path, backend="bm25", b=0)
    assert [e.passage_id for e in normalized.search("pump", minimum_score=0)] == ["p1", "p0"]
    tied = unnormalized.search("pump", minimum_score=0)
    assert tied[0].relevance == tied[1].relevance
    assert [e.passage_id for e in tied] == ["p0", "p1"]


def test_repeated_document_terms_help_but_saturate(tmp_path):
    path = corpus(tmp_path, ["pump", "pump pump", "pump " * 10])
    retriever = Retriever(path, backend="bm25", b=0)
    found = retriever.search("pump", minimum_score=0)
    assert [e.passage_id for e in found] == ["p2", "p1", "p0"]
    scores = {e.passage_id: e.relevance for e in found}
    assert 1 < scores["p1"] / scores["p0"] < 2
    assert scores["p2"] < 2.2 * scores["p0"]


def test_query_repetition_case_and_punctuation_do_not_inflate_scores(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["pump pressure", "motor"]), backend="bm25")
    assert retriever.search("PUMP, pump pump!", minimum_score=0) == retriever.search(
        "pump", minimum_score=0
    )


@pytest.mark.parametrize("query", ["", "   ", "?!", "the and or", "astronomy", "pumping"])
@pytest.mark.parametrize("backend", ["tfidf", "bm25"])
def test_empty_or_no_exact_token_overlap_always_abstains(tmp_path, query, backend):
    retriever = Retriever(corpus(tmp_path, ["pump pressure", "motor"]), backend=backend)
    assert retriever.search(query) == []
    assert retriever.search(query, minimum_score=0) == []


def test_bm25_empty_token_corpus_abstains(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["", "the and or"]), backend="bm25")
    assert retriever.search("pump", minimum_score=0) == []


def test_ties_keep_corpus_order_and_limit(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["pump"] * 6), backend="bm25")
    for _ in range(3):
        assert [e.passage_id for e in retriever.search("pump", limit=3, minimum_score=0)] == [
            "p0",
            "p1",
            "p2",
        ]


def test_bm25_has_an_independent_default_threshold(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["pump"] * 7), backend="bm25")
    assert retriever.minimum_score == 0.5
    assert retriever.search("pump") == []
    assert len(retriever.search("pump", minimum_score=0)) == 3


@pytest.mark.parametrize(
    "parameters",
    [
        {"backend": "unknown"},
        *(
            {"backend": "bm25", "k1": value}
            for value in [0, -1, float("nan"), float("inf"), True, "1"]
        ),
        *(
            {"backend": "bm25", "b": value}
            for value in [-0.1, 1.1, float("nan"), float("inf"), True, "1"]
        ),
    ],
)
def test_invalid_backend_or_bm25_parameters_rejected(tmp_path, parameters):
    with pytest.raises(ValueError):
        Retriever(corpus(tmp_path, ["pump"]), **parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"limit": 0},
        {"limit": 6},
        {"limit": 1.5},
        {"limit": True},
        *({"minimum_score": value} for value in [-1, float("nan"), float("inf"), True, "0"]),
    ],
)
def test_invalid_search_parameters_rejected(tmp_path, parameters):
    retriever = Retriever(corpus(tmp_path, ["pump"]), backend="bm25")
    with pytest.raises(ValueError):
        retriever.search("pump", **parameters)


def test_query_budget_is_preserved(tmp_path):
    retriever = Retriever(corpus(tmp_path, ["pump"]), backend="bm25")
    with pytest.raises(ValueError, match="budget"):
        retriever.search("pump " * 401)


def test_default_tfidf_rankings_and_scores_are_unchanged():
    path = ROOT / "knowledge/playbook.json"
    passages = json.loads(path.read_text())["passages"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(
        [" ".join([p["title"], *p["sensors"], p["text"], p["check"]]) for p in passages]
    )
    retriever = Retriever(path)
    assert retriever.backend == "tfidf"
    assert retriever.minimum_score == 0.12
    suite = json.loads((ROOT / "knowledge/retrieval-cases.json").read_text())
    for case in suite["cases"]:
        scores = (matrix @ vectorizer.transform([case["query"]]).T).toarray().ravel()
        expected = [i for i in np.argsort(-scores, kind="stable")[:3] if scores[i] >= 0.12]
        actual = retriever.search(case["query"])
        assert [e.passage_id for e in actual] == [passages[i]["id"] for i in expected]
        assert [e.relevance for e in actual] == pytest.approx([scores[i] for i in expected])


@pytest.mark.parametrize(
    "field,value",
    [
        ("passage_id", "invented"),
        ("title", "forged title"),
        ("text", "run commands"),
        ("check", "forged check"),
        ("version", "old"),
        ("corpus_sha256", "0" * 64),
    ],
)
def test_bm25_rejects_forged_evidence(field, value):
    retriever = Retriever(ROOT / "knowledge/playbook.json", backend="bm25")
    evidence = retriever.search("flow pressure")[0]
    assert retriever.verifies(evidence)
    assert not retriever.verifies(replace(evidence, **{field: value}))


def test_bm25_forged_evidence_cannot_reach_investigation():
    class ForgedRetriever(Retriever):
        def search(self, query, **kwargs):
            return [replace(super().search("pressure")[0], text="invented instruction")]

    training = pd.DataFrame({name: np.linspace(0, 10, 100) for name in FEATURES})
    frame = training.iloc[:8].copy()
    frame["datetime"] = pd.date_range("2020-01-01", periods=8, freq="s")
    frame["Pressure"] = 100
    metadata = {
        "threshold": 3.0,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
        "run_id": "synthetic-test",
        "model_sha256": "0" * 64,
    }
    report = investigate(
        frame,
        Detector.fit(training),
        metadata,
        "synthetic",
        "1" * 64,
        retriever=ForgedRetriever(ROOT / "knowledge/playbook.json", backend="bm25"),
    )
    assert report.status == "grounding_failed"
    assert not report.evidence and not report.suggested_checks


def test_evaluation_reports_rank_recall_mrr_abstention_and_verification(tmp_path):
    from forge.rag.evaluate import evaluate_retrieval

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    corpus(knowledge, ["pump pump", "pump motor", "bearing"])
    (knowledge / "retrieval-cases.json").write_text(
        json.dumps(
            {
                "purpose": "Synthetic metric check",
                "cases": [
                    {"query": "pump", "expected": ["p1"]},
                    {"query": "astronomy", "expected": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_retrieval(tmp_path, backend="bm25", minimum_score=0, b=0)
    assert result["backend"] == "bm25" and result["minimum_score"] == 0
    assert result["recall_at_3"] == 1 and result["mrr"] == 0.5
    assert result["correct_abstentions"] == result["expected_abstentions"] == 1
    assert result["citations_verified"]
    assert result["verified_citations"] == result["retrieved_citations"] == 2
    assert result["results"][0]["retrieved"] == ["p0", "p1"]
    assert result["results"][0]["reciprocal_rank"] == 0.5
    assert result["results"][1]["reciprocal_rank"] is None


def test_unchanged_nine_cases_compare_both_backends():
    from forge.rag.evaluate import evaluate_retrieval

    original = json.loads((ROOT / "knowledge/retrieval-cases.json").read_text())
    for backend in ["tfidf", "bm25"]:
        result = evaluate_retrieval(ROOT, backend=backend)
        assert result["recall_at_3"] == result["mrr"] == 1
        assert result["correct_abstentions"] == result["expected_abstentions"] == 2
        assert result["citations_verified"]
        assert len(result["results"]) == 9
        assert [
            {"query": c["query"], "expected": c["expected"]} for c in result["results"]
        ] == original["cases"]


def test_compare_cli_is_portable_and_records_separate_thresholds(tmp_path, monkeypatch, capsys):
    from forge.rag import evaluate

    monkeypatch.chdir(tmp_path)
    assert hasattr(evaluate, "main"), "A portable comparison module CLI is required"
    assert evaluate.main(["--root", str(ROOT), "--compare"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backends"]["tfidf"]["minimum_score"] == 0.12
    assert result["backends"]["bm25"]["minimum_score"] == 0.5
    assert result["backends"]["bm25"]["parameters"] == {"k1": 1.2, "b": 0.75}


@pytest.mark.parametrize("query,abstained", [("flow pressure", False), ("astronomy", True)])
def test_query_cli_exposes_verified_evidence(query, abstained, capsys):
    from forge.rag import evaluate

    assert hasattr(evaluate, "main"), "A query demo module CLI is required"
    assert evaluate.main(["--root", str(ROOT), "--backend", "bm25", "--query", query]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backend"] == "bm25" and result["minimum_score"] == 0.5
    assert result["abstained"] is abstained
    assert all(e["verified"] and e["citation"].endswith("@1.0") for e in result["results"])


def test_service_selects_bm25_and_retains_tfidf_default(monkeypatch):
    from forge.agents import service

    training = pd.DataFrame({name: np.linspace(0, 10, 100) for name in FEATURES})
    frame = training.iloc[:8].copy()
    frame["datetime"] = pd.date_range("2020-01-01", periods=8, freq="s")
    frame["Pressure"] = 100
    metadata = {
        "threshold": 3.0,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
        "run_id": "synthetic-test",
        "model_sha256": "0" * 64,
    }
    # Only replace disk model/recording loading. The workflow and retrieval are real.
    monkeypatch.setattr(
        service, "load_model", lambda *a, **kw: (Detector.fit(training), metadata, None)
    )
    monkeypatch.setattr(
        service, "load_development_recording", lambda *a, **kw: (frame, {"sha256": "1" * 64})
    )
    default = service.investigate_recording("synthetic", root=ROOT)
    explicit = service.investigate_recording("synthetic", root=ROOT, retrieval_backend="tfidf")
    bm25 = service.investigate_recording("synthetic", root=ROOT, retrieval_backend="bm25")
    assert default.evidence == explicit.evidence
    assert bm25.status == "needs_review" and bm25.review_status == "unreviewed"
    assert default.observation == bm25.observation
    assert default.evidence != bm25.evidence
    assert "flow-context@1.0" in {e.citation for e in bm25.evidence}


@pytest.mark.parametrize("k1", [1e-300, 1.7e308])
def test_extreme_finite_k1_cannot_create_nonfinite_evidence(tmp_path, k1):
    path = corpus(tmp_path, ["pump pump pressure", "pressure"])
    with np.errstate(over="raise", invalid="raise"):
        retriever = Retriever(path, backend="bm25", k1=k1)
        found = retriever.search("pump", minimum_score=0)
    assert len(found) == 1
    assert math.isfinite(found[0].relevance) and found[0].relevance > 0
