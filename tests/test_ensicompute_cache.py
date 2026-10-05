"""Exercise cache reuse, preservation and cleanup without a model or remote access."""

import os
import runpy
import shutil
import stat
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

worker = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ensicompute_ollama.py"))
globals_ = worker["run"].__globals__


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setitem(globals_, "TMP_ROOT", tmp_path)
    monkeypatch.setattr(os, "getuid", lambda: tmp_path.stat().st_uid, raising=False)
    monkeypatch.setattr("signal.SIGHUP", 1, raising=False)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setattr(os, "killpg", lambda *args: None, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)

    # Windows cannot exercise Linux ownership/mode semantics; tested separately below.
    def directory(path):
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    monkeypatch.setitem(globals_, "private_directory", directory)
    monkeypatch.setitem(globals_, "cache_lock", lambda cache: nullcontext())
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Unexpected download"))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: pytest.fail("Unexpected model server"))
    return worker["cache_directory"]()


def seed_cache(cache):
    (cache / "runtime/bin").mkdir(parents=True)
    binary = cache / "runtime/bin/ollama"
    binary.write_text("fake executable")
    binary.chmod(0o755)
    (cache / "runtime/.forge-complete").write_text("fixture")
    (cache / "models").mkdir()
    (cache / "models/blob").write_bytes(b"model contents")
    return binary


@pytest.mark.parametrize("shutdown_failure", [False, True])
def test_two_warm_sessions_keep_cache_without_downloads(sandbox, monkeypatch, shutdown_failure):
    cache = sandbox
    binary = seed_cache(cache)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    # Less than 12 GiB is fine when nothing needs downloading.
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=512 * 1024**2))

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 11435)

    monkeypatch.setattr("socket.socket", Socket)
    monkeypatch.setitem(
        globals_,
        "api",
        lambda port, path: (
            {"version": "test"} if path == "/api/version" else {"models": [{"name": "qwen3.5:4b"}]}
        ),
    )
    starts = []

    class Server:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    def popen(command, **kwargs):
        starts.append(command)
        assert command == [str(binary), "serve"]
        assert kwargs["env"]["OLLAMA_MODELS"] == str(cache / "models")
        return Server()

    monkeypatch.setattr("subprocess.Popen", popen)
    if shutdown_failure:

        def fail_shutdown(server):
            raise OSError("Process signalling failed")

        monkeypatch.setitem(globals_, "stop_server", fail_shutdown)
    for job in ("123", "124"):
        monkeypatch.setenv("SLURM_JOB_ID", job)
        monkeypatch.setattr("time.sleep", lambda seconds: worker["stop_session"](job))
        if shutdown_failure:
            with pytest.raises(OSError, match="signalling"):
                worker["run"]()
        else:
            worker["run"]()
        assert not (cache / "sessions" / job).exists()
        assert binary.is_file()
        assert (cache / "models/blob").read_bytes() == b"model contents"
    assert len(starts) == 2


def test_model_is_pulled_only_when_absent(sandbox, monkeypatch):
    responses = iter([{"models": []}, {"models": [{"name": "qwen3.5:4b"}]}])
    monkeypatch.setitem(globals_, "api", lambda *args: next(responses))
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=6 * 1024**3))
    calls = []
    monkeypatch.setattr("subprocess.run", lambda command, **kw: calls.append(command))
    worker["ensure_model"]("ollama", "qwen3.5:4b", 11435, {}, sandbox)
    assert calls == [["ollama", "pull", "qwen3.5:4b"]]


def test_incomplete_runtime_is_not_reused(sandbox):
    (sandbox / "runtime").mkdir()
    with pytest.raises(RuntimeError, match="Incomplete runtime"):
        worker["ensure_runtime"](sandbox, {})


def test_failed_install_removes_staging_but_not_models(sandbox, monkeypatch):
    (sandbox / "models").mkdir()
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr("shutil.which", lambda name: name)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        worker["ensure_runtime"](sandbox, {})
    assert not list(sandbox.glob("install-*"))
    assert not (sandbox / "runtime").exists()
    assert (sandbox / "models").is_dir()


def test_preserve_hardlinks_survive_old_directory_removal(sandbox, monkeypatch):
    old = sandbox.parent / "forge-ollama-old"
    old.mkdir()
    binary = seed_cache(old)
    # The legacy runtime has no completion marker.
    (old / "runtime/.forge-complete").unlink()
    monkeypatch.setenv("SLURM_JOB_ID", "59877")
    monkeypatch.setitem(globals_, "find_live_source", lambda job: old)
    monkeypatch.setitem(globals_, "api", lambda *args: {"models": [{"name": "qwen3.5:4b"}]})
    worker["preserve"](59877)
    assert os.path.samefile(binary, sandbox / "runtime/bin/ollama")
    assert os.path.samefile(old / "models/blob", sandbox / "models/blob")
    assert old.resolve().is_relative_to(sandbox.parent.resolve())
    shutil.rmtree(old)
    assert (sandbox / "models/blob").read_bytes() == b"model contents"
    assert (sandbox / "runtime/bin/ollama").read_text() == "fake executable"


def test_preservation_does_not_overwrite_cache(sandbox, monkeypatch):
    seed_cache(sandbox)
    monkeypatch.setenv("SLURM_JOB_ID", "59877")
    monkeypatch.setitem(globals_, "find_live_source", lambda job: sandbox)
    monkeypatch.setitem(globals_, "api", lambda *args: {"models": [{"name": "qwen3.5:4b"}]})
    with pytest.raises(RuntimeError, match="will not overwrite"):
        worker["preserve"](59877)


def test_stop_cannot_target_another_allocation(sandbox, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="does not match"):
        worker["stop_session"]("124")


@pytest.mark.parametrize(
    "mode,uid",
    [
        (stat.S_IFLNK | 0o700, 123),
        (stat.S_IFDIR | 0o777, 123),
        (stat.S_IFDIR | 0o700, 456),
    ],
)
def test_private_directory_rejects_symlink_permissions_or_wrong_owner(monkeypatch, mode, uid):
    monkeypatch.setattr(os, "getuid", lambda: 123, raising=False)
    path = SimpleNamespace(
        mkdir=lambda **kwargs: None, lstat=lambda: SimpleNamespace(st_mode=mode, st_uid=uid)
    )
    with pytest.raises(RuntimeError, match="Refusing"):
        worker["private_directory"](path)


def test_cache_lock_fails_busy_and_closes_descriptor(tmp_path, monkeypatch):
    seen = []

    def busy(fd, flags):
        seen.append(fd)
        raise BlockingIOError

    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=busy, LOCK_EX=2, LOCK_NB=4))
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    with pytest.raises(RuntimeError, match="already in use"):
        with worker["cache_lock"](tmp_path):
            pytest.fail("Acquired busy cache")
    with pytest.raises(OSError):
        os.fstat(seen[0])


def test_dead_server_parent_does_not_skip_runner_group_cleanup(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append(pid), raising=False)
    server = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda timeout: 0)
    worker["stop_server"](server)
    assert calls == [123]


def test_completed_install_publishes_once_and_is_reused(sandbox, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr("shutil.which", lambda name: name)
    calls = []

    def run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "tar":
            unpacked = Path(command[-1])
            (unpacked / "bin").mkdir()
            binary = unpacked / "bin/ollama"
            binary.write_text("fake executable")
            binary.chmod(0o755)

    monkeypatch.setattr("subprocess.run", run)
    first = worker["ensure_runtime"](sandbox, {})
    monkeypatch.setattr(
        "shutil.disk_usage", lambda path: pytest.fail("Warm runtime checked download space")
    )
    second = worker["ensure_runtime"](sandbox, {})
    assert first == second and calls == ["curl", "tar"]
    assert (sandbox / "runtime/.forge-complete").is_file()
    assert not list(sandbox.glob("install-*"))
