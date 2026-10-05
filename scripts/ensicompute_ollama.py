"""Compute-node Ollama cache and a separately owned, temporary server session."""

import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ARCHIVE = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
TMP_ROOT = Path("/tmp")
MODEL = "qwen3.5:4b"


def private_directory(path):
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"Refusing non-private or unowned directory: {path}")
    return path


def cache_directory():
    return private_directory(TMP_ROOT / f"forge-cache-{os.getuid()}")


@contextmanager
def cache_lock(cache):
    # Advisory kernel locks are released even if the supervisor is killed.
    import fcntl

    fd = os.open(cache / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "This node's FORGE cache is already in use by another session"
            ) from error
        yield
    finally:
        os.close(fd)


def require_space(path, gib):
    free = shutil.disk_usage(path).free / 1024**3
    if free < gib:
        raise RuntimeError(f"Need {gib:g} GiB free for this setup step; {free:.1f} GiB available")


def environment(folder, port, cache=None):
    cache = folder if cache is None else cache
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(folder),
            "TMPDIR": str(folder),
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(cache / "models"),
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_CONTEXT_LENGTH": "4096",
            "OLLAMA_KEEP_ALIVE": "30s",
        }
    )
    return env


def api(port, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
        return json.load(response)


def wait_ready(server, port):
    for _ in range(30):
        if server.poll() is not None:
            raise RuntimeError("Ollama exited before becoming ready")
        try:
            return api(port, "/api/version")
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("Ollama did not become ready within 30 checks")


def ensure_runtime(cache, env):
    runtime = cache / "runtime"
    marker = runtime / ".forge-complete"
    binary = runtime / "bin/ollama"
    if marker.is_file() and binary.is_file() and os.access(binary, os.X_OK):
        print(f"Reusing cached Ollama runtime: {runtime}", flush=True)
        return str(binary)
    if runtime.exists() or runtime.is_symlink():
        raise RuntimeError(f"Incomplete runtime at {runtime}; inspect it before replacing it")
    require_space(cache, 12)
    for tool in ("curl", "tar", "zstd"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing compute-node dependency: {tool}")
    stage = Path(tempfile.mkdtemp(prefix="install-", dir=cache))
    try:
        archive = stage / "ollama.tar.zst"
        unpacked = stage / "runtime"
        unpacked.mkdir()
        print("Downloading Ollama runtime into reusable node cache...", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--max-time",
                "600",
                "--output",
                str(archive),
                ARCHIVE,
            ],
            check=True,
            timeout=620,
            env=env,
        )
        subprocess.run(
            [
                "tar",
                "--zstd",
                "--no-same-owner",
                "-xf",
                str(archive),
                "-C",
                str(unpacked),
            ],
            check=True,
            timeout=180,
            env=env,
        )
        installed = unpacked / "bin/ollama"
        if not installed.is_file() or not os.access(installed, os.X_OK):
            raise RuntimeError("Runtime archive did not contain an executable Ollama binary")
        (unpacked / ".forge-complete").write_text(ARCHIVE, encoding="utf-8")
        unpacked.rename(runtime)
    finally:
        shutil.rmtree(stage)
    return str(binary)


def ensure_model(binary, model, port, env, cache):
    def installed():
        return any(item.get("name") == model for item in api(port, "/api/tags").get("models", []))

    if installed():
        print(f"Reusing cached model: {model}; no pull requested.", flush=True)
        return
    require_space(cache, 5)
    print(f"Downloading missing model {model} into reusable cache...", flush=True)
    subprocess.run([binary, "pull", model], env=env, check=True, timeout=900)
    if not installed():
        raise RuntimeError("Downloaded model was not found in Ollama's model list")


def current_job():
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise RuntimeError("This action requires an existing Slurm allocation")
    return job


def stop_session(job_id):
    if str(job_id) != current_job():
        raise RuntimeError("Stop request does not match the current Slurm allocation")
    folder = cache_directory() / "sessions" / str(job_id)
    if not folder.exists():
        print("No active cache session to stop.", flush=True)
        return
    private_directory(folder.parent)
    private_directory(folder)
    fd = os.open(folder / "stop", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    print(f"Requested cooperative stop for job {job_id}.", flush=True)


def stop_server(server):
    if server is None:
        return
    # The runner may still exist even when its parent server has already exited.
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=5)


def run(model=MODEL, port=11435):
    job = current_job()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Ollama requires a Slurm step with an assigned GPU")
    if model != MODEL or not 1024 <= port <= 65535:
        raise ValueError("Unsupported demo model or invalid port")
    cache = cache_directory()
    with cache_lock(cache):
        require_space(cache, 0.25)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        sessions = private_directory(cache / "sessions")
        folder = sessions / job
        # Never reuse a previous session folder or silently sweep other jobs.
        folder.mkdir(mode=0o700)
        private_directory(cache / "models")
        server = None
        try:
            env = environment(folder, port, cache)
            binary = ensure_runtime(cache, env)
            print(f"Reusable node cache: {cache}", flush=True)
            with (folder / "ollama.log").open("w", encoding="utf-8") as log:
                server = subprocess.Popen(
                    [binary, "serve"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                version = wait_ready(server, port)
                print(f"Ollama responding: {version}", flush=True)
                ensure_model(binary, model, port, env, cache)
                if server.poll() is not None:
                    raise RuntimeError("Ollama stopped during model setup")
                print(
                    json.dumps(
                        {
                            "event": "ollama_ready",
                            "node": socket.gethostname(),
                            "job_id": job,
                            "port": port,
                            "model": model,
                            "cache": str(cache),
                        }
                    ),
                    flush=True,
                )
                while not (folder / "stop").exists():
                    if server.poll() is not None:
                        raise RuntimeError("Ollama server exited unexpectedly")
                    time.sleep(1)
        except Exception:
            logfile = folder / "ollama.log"
            if logfile.exists():
                with logfile.open("rb") as log:
                    log.seek(max(0, logfile.stat().st_size - 3000))
                    print(log.read().decode("utf-8", errors="replace"), file=sys.stderr)
            raise
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_server(server)
            finally:
                private_directory(folder)
                if folder.parent != cache / "sessions" or folder.name != job:
                    raise RuntimeError("Refusing cleanup outside the current session")
                shutil.rmtree(folder)
                print(f"Removed session files; retained runtime and models at {cache}.", flush=True)


def find_live_source(job_id, proc=Path("/proc")):
    candidates = set()
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            raw = (process / "environ").read_bytes().split(b"\0")
            env = dict(item.split(b"=", 1) for item in raw if b"=" in item)
            if env.get(b"SLURM_JOB_ID") != str(job_id).encode():
                continue
            source = Path(os.fsdecode(env.get(b"HOME", b"")))
            if source.parent != TMP_ROOT or not source.name.startswith("forge-ollama-"):
                continue
            if env.get(b"OLLAMA_MODELS") != os.fsencode(source / "models"):
                continue
            candidates.add(source)
        except (OSError, ValueError):
            continue
    if len(candidates) != 1:
        raise RuntimeError("Could not identify one live legacy Ollama directory in this job")
    return private_directory(candidates.pop())


def link_tree(source, destination):
    # Runtime libraries may use symlinks; only allow links contained within the tree.
    root = source.resolve()
    for path in source.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Refusing external symbolic link: {path}")
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def preserve(job_id, port=11435):
    if str(job_id) != current_job():
        raise RuntimeError("Preservation request does not match the current allocation")
    source = find_live_source(job_id)
    if not any(m.get("name") == MODEL for m in api(port, "/api/tags").get("models", [])):
        raise RuntimeError("Current Ollama does not list the completed demo model")
    cache = cache_directory()
    with cache_lock(cache):
        if (cache / "runtime").exists() or (cache / "models").exists():
            raise RuntimeError("Cache already contains files; preservation will not overwrite them")
        stage = Path(tempfile.mkdtemp(prefix="preserve-", dir=cache))
        try:
            link_tree(source / "runtime", stage / "runtime")
            link_tree(source / "models", stage / "models")
            binary = stage / "runtime/bin/ollama"
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError("Legacy runtime is incomplete")
            (stage / "runtime/.forge-complete").write_text("Preserved from " + str(source))
            (stage / "runtime").rename(cache / "runtime")
            (stage / "models").chmod(0o700)
            (stage / "models").rename(cache / "models")
            print(
                f"Preserved runtime and model in {cache} using hard links; no download.", flush=True
            )
            print(
                "The old session remains running and still owns its original directory.", flush=True
            )
        finally:
            shutil.rmtree(stage)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        settings = json.loads(sys.argv[1])
        action = settings.pop("action", "run")
        if action == "stop":
            stop_session(**settings)
        elif action == "preserve":
            preserve(**settings)
        elif action == "run":
            run(**settings)
        else:
            raise ValueError("Unknown cache action")
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Ollama setup failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
