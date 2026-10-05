"""Offline checks for isolated storage, GPU assignment and server readiness."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

worker = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ensicompute_ollama.py"))


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Bootstrap tests must not download or start a model")

    monkeypatch.setattr("subprocess.Popen", forbidden)


def test_environment_preserves_assigned_gpu_and_uses_session_storage(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    folder = Path("/tmp/forge-ollama-example")
    env = worker["environment"](folder, 11435)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["OLLAMA_MODELS"] == str(folder / "models")
    assert env["HOME"] == str(folder)
    assert env["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert env["OLLAMA_NO_CLOUD"] == "1"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"


def test_no_gpu_allocation_stops_before_download(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        worker["run"]()


def test_low_disk_space_stops_before_download(monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=1024))
    with pytest.raises(RuntimeError, match="12 GiB"):
        worker["require_space"](Path("/tmp"), 12)


def test_dead_server_is_not_reported_ready():
    with pytest.raises(RuntimeError, match="exited"):
        worker["wait_ready"](SimpleNamespace(poll=lambda: 1), 11435)


def test_readiness_retries_then_succeeds(monkeypatch):
    calls = []

    def api(port, path):
        calls.append((port, path))
        if len(calls) == 1:
            raise OSError("Not listening yet")
        return {"version": "test"}

    monkeypatch.setitem(worker["wait_ready"].__globals__, "api", api)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    assert worker["wait_ready"](SimpleNamespace(poll=lambda: None), 11435) == {"version": "test"}
    assert len(calls) == 2


def test_readiness_eventually_times_out(monkeypatch):
    calls = []

    def api(port, path):
        calls.append(path)
        raise OSError("Not listening")

    monkeypatch.setitem(worker["wait_ready"].__globals__, "api", api)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="30 checks"):
        worker["wait_ready"](SimpleNamespace(poll=lambda: None), 11435)
    assert len(calls) == 30
