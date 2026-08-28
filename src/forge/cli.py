"""Development commands; no model calls or dataset downloads."""

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
    "langchain-openai",
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
        "stage": "foundation; no trained model or investigation pipeline yet",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FORGE development workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the interpreter and installed stack")
    app = commands.add_parser("app", help="Start the local foundation dashboard")
    app.add_argument("--port", type=int, default=8511)
    args = parser.parse_args()
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
