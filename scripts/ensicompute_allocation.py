"""Remote allocation supervisor, executed on nash through one SSH connection."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def supervise(resources, queue_seconds=300, ollama=None):
    """Own one uniquely named batch job and request its cancellation on exit."""
    name = "forge-session-" + uuid4().hex
    job_id = None
    step = None
    attempted = False
    exit_code = 0
    print(f"Session job name: {name}", flush=True)
    print(
        "Ollama will start after allocation; relay not started yet."
        if ollama
        else "Allocation only: Ollama and the relay are not started yet.",
        flush=True,
    )
    try:
        attempted = True
        output = command(
            [
                "sbatch",
                "--parsable",
                f"--job-name={name}",
                "--output=/dev/null",
                "--error=/dev/null",
                "--chdir=/tmp",
                *resources,
                "--wrap=exec sleep infinity",
            ]
        )
        # A cluster suffix would require cluster-aware monitoring and cleanup.
        if not re.fullmatch(r"[1-9][0-9]*", output):
            raise RuntimeError("Unexpected sbatch response; cancelling by session name.")
        job_id = output
        print(f"Submitted job {job_id}. Waiting up to {queue_seconds}s for a node.", flush=True)
        deadline = time.monotonic() + queue_seconds
        running = False
        previous = None
        while True:
            status = command(["squeue", "--noheader", f"--jobs={job_id}", "--format=%T|%N"])
            if not status:
                if not running:
                    raise RuntimeError(
                        "Job left the queue before a running allocation was observed."
                    )
                print("Job left the queue; the allocation is no longer active.", flush=True)
                break
            state, separator, node = status.partition("|")
            if not separator or "\n" in status:
                raise RuntimeError("Unexpected squeue response.")
            state, node = state.strip(), node.strip()
            if status != previous:
                print(f"Job {job_id}: {state}; node: {node or 'not assigned'}", flush=True)
                previous = status
            if state == "RUNNING" and not running:
                if not node or node in {"(null)", "N/A"}:
                    raise RuntimeError("Running allocation has no assigned node.")
                running = True
                print(
                    json.dumps({"event": "allocated", "job_id": job_id, "node": node}), flush=True
                )
                print(
                    "Keep this terminal open. Ctrl+C releases this session's allocation.",
                    flush=True,
                )
                if ollama:
                    step = subprocess.Popen(
                        [
                            "srun",
                            f"--jobid={job_id}",
                            "--overlap",
                            "--exact",
                            "--nodes=1",
                            "--ntasks=1",
                            "--cpus-per-task=1",
                            "--gres=shard:1",
                            "python3",
                            "-u",
                            "-c",
                            ollama["source"],
                            json.dumps({"model": ollama["model"], "port": ollama["port"]}),
                        ]
                    )
            if step is not None and step.poll() is not None:
                raise RuntimeError(f"Ollama compute step exited with code {step.returncode}")
            if not running and time.monotonic() >= deadline:
                raise RuntimeError("Queue wait exceeded the configured limit.")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping allocation session.", flush=True)
        exit_code = 130
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Allocation failed: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if attempted:
            # Do not let a second Ctrl+C or hangup interrupt the cleanup request.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            if step is not None and step.poll() is None:
                # Stop cooperatively before signalling srun, which can abort tasks quickly.
                try:
                    command(
                        [
                            "srun",
                            f"--jobid={job_id}",
                            "--overlap",
                            "--exact",
                            "--nodes=1",
                            "--ntasks=1",
                            "--cpus-per-task=1",
                            "--gres=none",
                            "python3",
                            "-c",
                            ollama["source"],
                            json.dumps({"action": "stop", "job_id": job_id}),
                        ]
                    )
                    step.wait(timeout=20)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    print(
                        f"Cooperative stop incomplete: {error}; signalling step.",
                        file=sys.stderr,
                        flush=True,
                    )
                    try:
                        step.terminate()
                        step.wait(timeout=15)
                    except (OSError, subprocess.TimeoutExpired) as stop_error:
                        print(
                            f"Compute-step shutdown incomplete: {stop_error}",
                            file=sys.stderr,
                            flush=True,
                        )
            selector = [job_id] if job_id else [f"--name={name}", f"--user={os.getuid()}"]
            try:
                command(["scancel", *selector])
                print(f"Cancellation requested for {job_id or name}.", flush=True)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                print(
                    f"Cleanup could not be confirmed: {error}. On nash, check squeue "
                    f"for {job_id or name} and cancel that job if it remains.",
                    file=sys.stderr,
                    flush=True,
                )
                exit_code = 1
    return exit_code


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    settings = json.loads(sys.argv[1])
    return supervise(settings["resources"], settings["queue_seconds"], settings.get("ollama"))


if __name__ == "__main__":
    raise SystemExit(main())
