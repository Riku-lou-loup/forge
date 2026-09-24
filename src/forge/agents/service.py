"""Application boundary: trusted model, pinned recordings, local evidence only."""

from forge.agents.workflow import Budget, investigate
from forge.config import PROJECT_ROOT
from forge.data.partitions import inspect_recording, read_plan, select_records
from forge.ml.training import load_model
from forge.rag.retrieval import Retriever


def available_recordings(root=PROJECT_ROOT):
    inventory, plan = read_plan(root)
    return [
        *select_records(inventory, plan, "train"),
        *select_records(inventory, plan, "validation"),
    ]


def load_development_recording(experiment_id, root=PROJECT_ROOT):
    record = next(
        (r for r in available_recordings(root) if r["experiment_id"] == experiment_id), None
    )
    if record is None:
        raise ValueError(
            "Choose a training or validation recording; test is reserved for explicit evaluation."
        )
    return inspect_recording(record, root), record


def investigate_recording(
    experiment_id="valve1/1", *, root=PROJECT_ROOT, budget=Budget(), retrieval_backend="tfidf"
):
    detector, metadata, _ = load_model(root, active=True)
    frame, record = load_development_recording(experiment_id, root)
    path = root / "knowledge/playbook.json"
    retriever = Retriever(path, backend=retrieval_backend) if path.exists() else None
    return investigate(
        frame,
        detector,
        metadata,
        experiment_id,
        record["sha256"],
        retriever=retriever,
        budget=budget,
    )
