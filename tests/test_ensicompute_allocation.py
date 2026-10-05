"""Exercise remote Slurm lifecycle with fake commands, never actual allocations."""

import runpy
import subprocess
from pathlib import Path

import pytest

worker = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/ensicompute_allocation.py")
)
supervise = worker["supervise"]


@pytest.fixture(autouse=True)
def isolate_remote_os(monkeypatch):
    monkeypatch.setattr("signal.SIGHUP", 1, raising=False)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setattr("os.getuid", lambda: 154310, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("No external processes permitted in lifecycle tests")

    monkeypatch.setattr("subprocess.run", forbidden)


def test_running_job_cancelled_on_interrupt(monkeypatch, capsys):
    calls = []

    def command(args):
        calls.append(args)
        return {"sbatch": "59623", "squeue": "RUNNING|ampere", "scancel": ""}[args[0]]

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setitem(supervise.__globals__, "command", command)
    monkeypatch.setattr("time.sleep", interrupt)
    assert supervise(["--partition=a40", "--time=60"]) == 130
    assert calls[-1] == ["scancel", "59623"]
    assert "--output=/dev/null" in calls[0] and "--time=60" in calls[0]
    assert '"node": "ampere"' in capsys.readouterr().out


def test_pending_job_cancelled_at_queue_deadline(monkeypatch):
    calls = []

    def command(args):
        calls.append(args)
        return {"sbatch": "123", "squeue": "PENDING|", "scancel": ""}[args[0]]

    clock = iter([0, 301])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setitem(supervise.__globals__, "command", command)
    assert supervise([]) == 1
    assert calls[-1] == ["scancel", "123"]


@pytest.mark.parametrize("outcome", ["timeout", "malformed", "rejected"])
def test_uncertain_submission_cleanup_uses_unique_session_name(monkeypatch, outcome):
    calls = []

    def command(args):
        calls.append(args)
        if args[0] == "sbatch":
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(args, 30)
            if outcome == "rejected":
                raise RuntimeError("Submission refused")
            return "not-a-job-id"
        return ""

    monkeypatch.setitem(supervise.__globals__, "command", command)
    assert supervise([]) == 1
    name = next(a.split("=", 1)[1] for a in calls[0] if a.startswith("--job-name="))
    assert name.startswith("forge-session-")
    assert calls[-1] == ["scancel", f"--name={name}", "--user=154310"]


def test_cleanup_failure_is_not_reported_as_success(monkeypatch, capsys):
    def command(args):
        if args[0] == "scancel":
            raise RuntimeError("Controller unreachable")
        return "123" if args[0] == "sbatch" else ""

    monkeypatch.setitem(supervise.__globals__, "command", command)
    assert supervise([]) == 1
    assert "Cleanup could not be confirmed" in capsys.readouterr().err


def test_finished_allocation_exits_without_waiting_forever(monkeypatch):
    states = iter(["RUNNING|ampere", ""])
    calls = []

    def command(args):
        calls.append(args)
        if args[0] == "squeue":
            return next(states)
        return "123" if args[0] == "sbatch" else ""

    monkeypatch.setitem(supervise.__globals__, "command", command)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    assert supervise([]) == 0
    assert calls[-1] == ["scancel", "123"]


def test_compute_step_stops_before_job_cancellation(monkeypatch):
    events = []

    class Step:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate_step")

        def wait(self, timeout):
            events.append("wait_step")
            self.returncode = 130

    def popen(args):
        assert "--jobid=123" in args and "--gres=shard:1" in args
        assert "--overlap" in args and "--exact" in args
        events.append("start_step")
        return Step()

    def command(args):
        if args[0] == "scancel":
            events.append("cancel_job")
        if args[0] == "srun":
            assert "--gres=none" in args
            assert '"action": "stop"' in args[-1]
            events.append("cooperative_stop")
            return ""
        return {"sbatch": "123", "squeue": "RUNNING|ampere", "scancel": ""}[args[0]]

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setitem(supervise.__globals__, "command", command)
    monkeypatch.setattr("time.sleep", interrupt)
    assert supervise([], ollama={"source": "pass", "model": "qwen3.5:4b", "port": 11435}) == 130
    assert events == ["start_step", "cooperative_stop", "wait_step", "cancel_job"]


def test_cooperative_stop_failure_falls_back_before_cancellation(monkeypatch):
    events = []

    class Step:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append("wait")
            self.returncode = 130

    def command(args):
        if args[0] == "srun":
            events.append("stop_failed")
            raise RuntimeError("Step unavailable")
        if args[0] == "scancel":
            events.append("cancel")
        return {"sbatch": "123", "squeue": "RUNNING|ampere", "scancel": ""}[args[0]]

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: Step())
    monkeypatch.setitem(supervise.__globals__, "command", command)
    monkeypatch.setattr("time.sleep", interrupt)
    assert supervise([], ollama={"source": "pass", "model": "qwen3.5:4b", "port": 11435}) == 130
    assert events == ["stop_failed", "terminate", "wait", "cancel"]
