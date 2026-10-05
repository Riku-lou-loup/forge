"""Preview EnsiCompute settings and optionally check SSH access to nash."""

import argparse
import base64
import json
import re
import shlex
import shutil
import subprocess
import sys
import zlib
from pathlib import Path


def parse_args(argv=None):
    """Read options and reject invalid settings before any remote work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--partition", choices=("a40", "rtx6000", "v100"), default="a40")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--local-port", type=int, default=11435)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-ssh",
        action="store_true",
        help="Connect to nash and run hostname; do not allocate a job",
    )
    mode.add_argument(
        "--allocate",
        action="store_true",
        help="Reserve a Slurm job until stopped",
    )
    mode.add_argument(
        "--connect-job",
        type=int,
        help="Attach a relay and tunnel to an existing job; do not allocate or cancel it",
    )
    mode.add_argument(
        "--preserve-job",
        type=int,
        help="Preserve a running legacy session's downloads in the reusable node cache",
    )
    parser.add_argument("--relay-port", type=int, default=11436)
    parser.add_argument("--queue-seconds", type=int, default=300)
    parser.add_argument(
        "--start-ollama",
        action="store_true",
        help="With --allocate, reuse the node cache or download missing Ollama/model files",
    )
    parser.add_argument("--remote-port", type=int, default=11435)
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", args.user):
        parser.error(
            "--user must be an account name using letters, digits, underscore, dot or dash"
        )
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    if args.memory_gb < 1:
        parser.error("--memory-gb must be positive")
    if not 1 <= args.minutes <= 240:
        parser.error("--minutes must be between 1 and 240")
    if not 1024 <= args.local_port <= 65535:
        parser.error("--local-port must be between 1024 and 65535")
    if not 1 <= args.queue_seconds <= 3600:
        parser.error("--queue-seconds must be between 1 and 3600")
    if args.start_ollama and not args.allocate:
        parser.error("--start-ollama requires --allocate")
    if not 1024 <= args.remote_port <= 65535:
        parser.error("--remote-port must be between 1024 and 65535")
    if args.connect_job is not None and args.connect_job <= 0:
        parser.error("--connect-job must be a positive job ID")
    if not 1024 <= args.relay_port <= 65535:
        parser.error("--relay-port must be between 1024 and 65535")
    if args.preserve_job is not None and args.preserve_job <= 0:
        parser.error("--preserve-job must be a positive job ID")
    return args


def build_allocation_command(args):
    """Build resource arguments from validated settings; no workload or submission yet."""
    return [
        "srun",
        f"--partition={args.partition}",
        "--gres=shard:1",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.memory_gb}G",
        f"--time={args.minutes}",
    ]


def check_ssh(user):
    """Let OpenSSH handle authentication; Python never reads a password."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", user):
        raise ValueError("Invalid SSH account name")
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH was not found. Enable the Windows OpenSSH Client first.")
    command = [
        executable,
        "-T",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=ask",
        f"{user}@nash.ensimag.fr",
        "hostname",
    ]
    print("Checking SSH. Enter your password in the SSH prompt if requested.", flush=True)
    try:
        # Inherit stdin/stderr for SSH prompts; capture only remote standard output.
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "SSH check timed out after 60 seconds. Check VPN and login access."
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH exited with code {result.returncode}. See the SSH message above; "
            "check VPN, authentication and host-key verification."
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or lines[-1].lower() not in {"nash", "nash.ensimag.fr"}:
        raise RuntimeError("SSH completed, but hostname did not identify nash.")
    return lines[-1]


def remote_python(source, arguments):
    """Compress code and arguments to stay within Windows command-line limits."""
    payload = json.dumps({"source": source, "args": arguments}).encode("utf-8")
    encoded = base64.b64encode(zlib.compress(payload)).decode("ascii")
    bootstrap = (
        "import base64,json,sys,zlib;"
        "p=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));"
        "sys.argv=['-c',*p['args']];"
        "exec(compile(p['source'],'<forge-remote>','exec'))"
    )
    return shlex.join(["python3", "-u", "-c", bootstrap, encoded])


def allocate_session(args):
    """Run the supervisor on nash; authentication and output stay in the terminal."""
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH was not found. Enable the Windows OpenSSH Client first.")
    worker = Path(__file__).with_name("ensicompute_allocation.py").read_text(encoding="utf-8")
    settings = json.dumps(
        {
            "resources": build_allocation_command(args)[1:],
            "queue_seconds": args.queue_seconds,
            "ollama": (
                {
                    "source": Path(__file__)
                    .with_name("ensicompute_ollama.py")
                    .read_text(encoding="utf-8"),
                    "model": "qwen3.5:4b",
                    "port": args.remote_port,
                }
                if args.start_ollama
                else None
            ),
        }
    )
    # SSH invokes a remote Linux shell: quote every argument for that shell.
    remote = remote_python(worker, [settings])
    command = [
        executable,
        "-tt",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=ask",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        f"{args.user}@nash.ensimag.fr",
        remote,
    ]
    print("Opening allocation session. Authenticate in SSH; Ctrl+C stops it.", flush=True)
    try:
        result = subprocess.run(command, check=False, shell=False)
    except KeyboardInterrupt:
        print(
            "\nSSH interrupted. Remote cleanup is best effort; check the printed job ID "
            "on nash if cancellation was not confirmed.",
            file=sys.stderr,
        )
        return 130
    if result.returncode == 130:
        print("Allocation session stopped by interrupt; see the remote cleanup result above.")
    elif result.returncode:
        print(
            "Allocation session ended with an error. If a job was submitted, "
            "check its printed ID/name on nash when cleanup was not confirmed.",
            file=sys.stderr,
        )
    return result.returncode


def connect_job(args):
    """Forward a local port to a relay on nash without owning the existing job."""
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH was not found. Enable the Windows OpenSSH Client first.")
    source = Path(__file__).with_name("ensicompute_relay.py").read_text(encoding="utf-8")
    remote = remote_python(
        source,
        [
            "--job-id",
            str(args.connect_job),
            "--listen-port",
            str(args.relay_port),
            "--upstream-port",
            str(args.remote_port),
        ],
    )
    command = [
        executable,
        "-tt",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=ask",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-L",
        f"127.0.0.1:{args.local_port}:127.0.0.1:{args.relay_port}",
        f"{args.user}@nash.ensimag.fr",
        remote,
    ]
    print(f"Connecting to existing job {args.connect_job}; authenticate in SSH.", flush=True)
    print(f"FORGE endpoint once relay is ready: http://127.0.0.1:{args.local_port}", flush=True)
    print(
        "Keep the allocation terminal open. Stopping this connection does not cancel that job.",
        flush=True,
    )
    try:
        result = subprocess.run(command, shell=False, check=False)
    except KeyboardInterrupt:
        print("\nConnection interrupted; the existing allocation was not cancelled.")
        return 130
    if result.returncode not in (0, 130):
        print("Relay connection failed; see SSH or relay output above.", file=sys.stderr)
    return result.returncode


def preserve_job(args):
    """Run a bounded, CPU-only preservation step in a live legacy allocation."""
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH was not found")
    source = Path(__file__).with_name("ensicompute_ollama.py").read_text(encoding="utf-8")
    settings = json.dumps(
        {"action": "preserve", "job_id": args.preserve_job, "port": args.remote_port}
    )
    step = shlex.join(
        [
            "srun",
            f"--jobid={args.preserve_job}",
            "--overlap",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--gres=none",
        ]
    )
    remote = step + " " + remote_python(source, [settings])
    command = [
        executable,
        "-tt",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=ask",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        f"{args.user}@nash.ensimag.fr",
        remote,
    ]
    print("Preserving the running session's completed downloads; authenticate in SSH.", flush=True)
    try:
        return subprocess.run(command, shell=False, check=False, timeout=180).returncode
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Preservation timed out; do not assume the cache is complete") from error
    except KeyboardInterrupt:
        print("Preservation interrupted; the existing allocation was not cancelled.")
        return 130


def main(argv=None):
    args = parse_args(argv)
    if args.preserve_job is not None:
        try:
            return preserve_job(args)
        except (OSError, RuntimeError) as error:
            print(f"Cache preservation failed: {error}", file=sys.stderr)
            return 1
    if args.connect_job is not None:
        try:
            return connect_job(args)
        except (OSError, RuntimeError) as error:
            print(f"Relay connection failed: {error}", file=sys.stderr)
            return 1
    if args.allocate:
        print("Allocation requested; settings below.")
    else:
        print("Configuration preview only; no connection or job has been created.")
    print(f"Account: {args.user}@nash.ensimag.fr")
    print(f"Partition: {args.partition} (Slurm will select the node)")
    print(f"CPUs: {args.cpus}; memory: {args.memory_gb} GiB; duration: {args.minutes} minutes")
    print(f"Local tunnel port: {args.local_port}")
    if args.start_ollama:
        print(
            "Ollama will reuse the node cache, downloading only missing files; relay not started."
        )
    print("Allocation arguments (preview; workload not added yet):")
    print(" ".join(build_allocation_command(args)))
    if args.allocate:
        try:
            return allocate_session(args)
        except (OSError, RuntimeError) as error:
            print(f"Allocation setup failed: {error}", file=sys.stderr)
            return 1
    if args.check_ssh:
        try:
            hostname = check_ssh(args.user)
        except (OSError, RuntimeError) as error:
            print(f"SSH check failed: {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nSSH check cancelled.", file=sys.stderr)
            return 130
        print(f"SSH verified: {hostname}. No Slurm job was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
