"""Run an explicitly enabled local Ollama investigation and export its trace."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from forge.agents.client_factory import create_client
from forge.agents.local_agent import investigate_local, markdown_report
from forge.agents.local_memory import require_memory_headroom
from forge.agents.ollama_client import OllamaError
from forge.agents.tools import InvestigationTools
from forge.config import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-llm", action="store_true", help="Allow calls to local Ollama")
    parser.add_argument("--recording", default="valve1/1")
    parser.add_argument(
        "--question", default="Investigate this recording. What happened and what should I check?"
    )
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--backend", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--backend-llm", choices=("ollama", "ensicompute"), default="ollama")
    parser.add_argument("--torch-artifact", type=Path)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args(argv)
    if not args.enable_llm:
        parser.error("Pass --enable-llm to authorize local model calls for this run.")
    client = None
    try:
        # If user explicitly selected Ollama, check that the system has enough free memory to run it.
        if args.backend_llm == "ollama":
            require_memory_headroom()
        client = create_client(args.backend_llm, model=args.model, port=args.port)
        print("Running the local investigation. Qwen will unload when it finishes.", flush=True)
        result = investigate_local(
            args.question,
            args.recording,
            enabled=True,
            client=client,
            tools=InvestigationTools(
                root=args.root, backend=args.backend, torch_artifact=args.torch_artifact
            ),
        )
        print(markdown_report(result))
        print(f"Local model: {result['model']['model']}; elapsed: {result['elapsed_seconds']} s")
        if args.export:
            folder = args.root / "reports/incidents"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromisoformat(result["generated_at"]).strftime("%Y%m%dT%H%M%S%fZ")
            recording = args.recording.replace("/", "-")
            stem = f"local-llm-{stamp}-{recording}-{uuid4().hex[:8]}"
            trace = folder / (stem + ".json")
            trace.write_text(json.dumps(result, indent=2), encoding="utf-8")
            (folder / (stem + ".md")).write_text(markdown_report(result), encoding="utf-8")
            print(f"Saved trace: {trace}")
        return 0 if result["status"] == "draft" else 2
    except (OSError, ValueError, OllamaError) as error:
        parser.error(str(error))
    finally:
        if client is not None:
            try:
                client.unload()
            except OllamaError:
                print(
                    "Could not confirm model unload. Check the local Ollama server.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
