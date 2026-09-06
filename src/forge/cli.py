"""Local training, evaluation, investigation, and application commands."""

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from forge.config import PROJECT_ROOT, Settings

DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "plotly",
    "streamlit",
    "langgraph",
    "langsmith",
    "langchain-openai",
    "pydantic",
    "pydantic-settings",
    "pypdf",
)


def environment_report() -> dict:
    packages = {}
    for package in DEPENDENCIES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    settings = Settings()
    return {
        "project": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "isolated_environment": sys.prefix != sys.base_prefix,
        "packages": packages,
        "model_configured": settings.model_configured,
        "stage": "anomaly detector and local evidence investigation",
        "trained_artifact_present": (PROJECT_ROOT / "models/latest.json").exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FORGE development workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the interpreter and installed stack")
    commands.add_parser("train", help="Fit on normal training rows and select on validation")
    evaluation = commands.add_parser("evaluate", help="Evaluate the frozen model on held-out data")
    evaluation.add_argument("--allow-test", action="store_true")
    investigation = commands.add_parser("investigate", help="Draft a local evidence report")
    investigation.add_argument(
        "--recording", default="valve1/1", help="Training or validation recording ID"
    )
    investigation.add_argument(
        "--export", action="store_true", help="Save an unreviewed JSON and Markdown draft"
    )
    app = commands.add_parser("app", help="Start the local investigation dashboard")
    app.add_argument("--port", type=int, default=8511)
    args = parser.parse_args()
    if args.command in {"train", "evaluate", "investigate"}:
        try:
            from forge.ml.training import evaluate_test, train

            if args.command == "train":
                folder, _ = train()
                print(f"Frozen model: {folder}")
            elif args.command == "evaluate":
                print(json.dumps(evaluate_test(allow_test=args.allow_test), indent=2))
            else:
                from forge.agents.service import investigate_recording
                from forge.reports.incident import export, markdown

                report = investigate_recording(args.recording)
                if args.export:
                    for path in export(report, PROJECT_ROOT / "reports/incidents"):
                        print(path)
                else:
                    print(markdown(report))
            return 0
        except (ValueError, OSError, KeyError) as error:
            print(f"FORGE: {error}", file=sys.stderr)
            return 2
    if args.command == "doctor":
        report = environment_report()
        print(json.dumps(report, indent=2))
        return 0 if all(report["packages"].values()) and report["isolated_environment"] else 1
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    app_path = Path(__file__).parent / "ui" / "app.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address=127.0.0.1",
            f"--server.port={args.port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        cwd=PROJECT_ROOT,
    )
