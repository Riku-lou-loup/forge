"""Exercise typed tools using a scripted caller, not an LLM."""

import argparse
import json
from pathlib import Path

from forge.agents.tool_runtime import ToolCall, ToolSession
from forge.agents.tools import InvestigationTools, tool_definitions
from forge.config import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--recording", default="valve1/1")
    parser.add_argument("--backend", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--torch-artifact", type=Path)
    parser.add_argument(
        "--schemas", action="store_true", help="Print schemas without loading resources"
    )
    args = parser.parse_args(argv)
    if args.schemas:
        print(json.dumps(tool_definitions(), indent=2))
        return 0
    try:
        session = ToolSession(
            InvestigationTools(
                root=args.root, backend=args.backend, torch_artifact=args.torch_artifact
            )
        )
        inspection = session.execute(
            ToolCall(
                call_id="inspect-1",
                name="inspect_recording",
                arguments={"recording_id": args.recording},
            )
        )
        if inspection.result is not None and inspection.result.status == "alert":
            query = " ".join(item.sensor for item in inspection.result.sensors)
            session.execute(
                ToolCall(call_id="search-1", name="search_evidence", arguments={"query": query})
            )
        print(
            json.dumps(
                {
                    "mode": "scripted_offline_tool_demo",
                    "llm_called": False,
                    "routing": "Fixed demo policy: inspect, then search only if an alert exists.",
                    "trace": [entry.model_dump(mode="json") for entry in session.trace],
                },
                indent=2,
            )
        )
        return 2 if any(item.error is not None for item in session.trace) else 0
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
