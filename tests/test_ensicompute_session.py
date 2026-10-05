"""Offline validation for the session configuration preview."""

import base64
import json
import runpy
import subprocess
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ensicompute_session.py"
session = runpy.run_path(str(SCRIPT))
parse_args = session["parse_args"]


def test_defaults():
    args = parse_args(["--user", "dangddk"])
    assert vars(args) == {
        "user": "dangddk",
        "partition": "a40",
        "cpus": 4,
        "memory_gb": 16,
        "minutes": 60,
        "local_port": 11435,
        "check_ssh": False,
        "allocate": False,
        "queue_seconds": 300,
        "start_ollama": False,
        "remote_port": 11435,
        "connect_job": None,
        "relay_port": 11436,
        "preserve_job": None,
    }


def test_overrides_and_valid_boundaries():
    args = parse_args(
        [
            "--user",
            "student",
            "--partition",
            "v100",
            "--cpus",
            "1",
            "--memory-gb",
            "1",
            "--minutes",
            "240",
            "--local-port",
            "65535",
        ]
    )
    assert (args.partition, args.cpus, args.memory_gb) == ("v100", 1, 1)
    assert (args.minutes, args.local_port) == (240, 65535)
    args = parse_args(["--user", "student", "--minutes", "1", "--local-port", "1024"])
    assert (args.minutes, args.local_port) == (1, 1024)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--cpus", "0"),
        ("--cpus", "-1"),
        ("--cpus", "four"),
        ("--memory-gb", "0"),
        ("--memory-gb", "-2"),
        ("--minutes", "0"),
        ("--minutes", "241"),
        ("--local-port", "1023"),
        ("--local-port", "65536"),
        ("--partition", "unknown"),
        ("--user", ""),
        ("--user", "two names"),
    ],
)
def test_invalid_settings_fail_clearly(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["--user", "student", option, value])
    assert error.value.code == 2
    assert option in capsys.readouterr().err


def test_user_is_required():
    with pytest.raises(SystemExit) as error:
        parse_args([])
    assert error.value.code == 2


def test_preview_does_not_run_commands(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not launch a process or connect")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    assert session["main"](["--user", "dangddk"]) == 0
    output = capsys.readouterr().out
    assert "no connection or job has been created" in output
    assert "dangddk@nash.ensimag.fr" in output
    assert "11435" in output


def test_ssh_uses_fixed_command_and_inherits_authentication(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "Welcome\nnash.ensimag.fr\n")

    monkeypatch.setattr("subprocess.run", run)
    assert session["check_ssh"]("dangddk") == "nash.ensimag.fr"
    command, kwargs = calls[0]
    assert command[-2:] == ["dangddk@nash.ensimag.fr", "hostname"]
    assert "StrictHostKeyChecking=ask" in command
    assert "NumberOfPasswordPrompts=1" in command
    assert kwargs["shell"] is False and kwargs["timeout"] == 60
    assert "input" not in kwargs and "stdin" not in kwargs and "stderr" not in kwargs


@pytest.mark.parametrize("failure", ["missing", "timeout", "denied", "wrong_host", "empty"])
def test_ssh_failure_returns_nonzero(monkeypatch, failure, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None if failure == "missing" else "ssh.exe")

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        return subprocess.CompletedProcess(
            command,
            255 if failure == "denied" else 0,
            "ampere\n" if failure == "wrong_host" else "",
        )

    monkeypatch.setattr("subprocess.run", run)
    assert session["main"](["--user", "dangddk", "--check-ssh"]) == 1
    assert "SSH check failed" in capsys.readouterr().err


@pytest.mark.parametrize("user", ["-oProxyCommand=x", "name@otherhost", "name;hostname"])
def test_ssh_rejects_invalid_user_before_launch(monkeypatch, user):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("SSH launched"))
    with pytest.raises(ValueError, match="account"):
        session["check_ssh"](user)


def test_ssh_cancel_is_reported(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.run", cancel)
    assert session["main"](["--user", "dangddk", "--check-ssh"]) == 130


def test_allocation_defaults_do_not_launch_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Building allocation arguments must not launch a process")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    args = parse_args(["--user", "dangddk"])
    assert session["build_allocation_command"](args) == [
        "srun",
        "--partition=a40",
        "--gres=shard:1",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=60",
    ]


@pytest.mark.parametrize("partition", ["a40", "rtx6000", "v100"])
def test_allocation_uses_selected_resources(partition):
    args = parse_args(
        [
            "--user",
            "dangddk",
            "--partition",
            partition,
            "--cpus",
            "2",
            "--memory-gb",
            "8",
            "--minutes",
            "30",
        ]
    )
    command = session["build_allocation_command"](args)
    assert f"--partition={partition}" in command
    assert "--cpus-per-task=2" in command
    assert "--mem=8G" in command and "--time=30" in command
    assert not any(arg.startswith("--nodelist") for arg in command)


def test_preview_displays_unsubmitted_allocation(capsys):
    assert session["main"](["--user", "dangddk"]) == 0
    output = capsys.readouterr().out
    assert "preview; workload not added yet" in output
    assert "srun --partition=a40 --gres=shard:1" in output


def test_allocation_uses_one_ssh_session(monkeypatch):
    import json
    import shlex

    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", run)
    assert session["main"](["--user", "dangddk", "--allocate"]) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "-tt" in command and "StrictHostKeyChecking=ask" in command
    assert command[-2] == "dangddk@nash.ensimag.fr"
    remote = shlex.split(command[-1])
    assert remote[:3] == ["python3", "-u", "-c"]
    packed = json.loads(zlib.decompress(base64.b64decode(remote[4])))
    compile(packed["source"], "remote_worker", "exec")
    assert (
        json.loads(packed["args"][0])["resources"]
        == session["build_allocation_command"](parse_args(["--user", "dangddk"]))[1:]
    )
    assert kwargs == {"check": False, "shell": False}


def test_allocation_and_check_modes_are_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--user", "dangddk", "--allocate", "--check-ssh"])


@pytest.mark.parametrize("seconds", ["0", "3601"])
def test_queue_wait_is_bounded(seconds):
    with pytest.raises(SystemExit):
        parse_args(["--user", "dangddk", "--queue-seconds", seconds])


@pytest.mark.parametrize("returncode", [0, 130, 1, 255])
def test_allocation_exit_messages_distinguish_interrupt_from_failure(
    monkeypatch, capsys, returncode
):
    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, returncode),
    )
    assert session["main"](["--user", "dangddk", "--allocate"]) == returncode
    output = capsys.readouterr()
    if returncode == 130:
        assert "stopped by interrupt" in output.out
        assert "ended with an error" not in output.err
    elif returncode:
        assert "ended with an error" in output.err
    else:
        assert not output.err


def test_ollama_start_requires_allocation():
    with pytest.raises(SystemExit):
        parse_args(["--user", "dangddk", "--start-ollama"])


@pytest.mark.parametrize("port", ["0", "65536"])
def test_remote_port_is_validated(port):
    with pytest.raises(SystemExit):
        parse_args(["--user", "dangddk", "--remote-port", port])


def test_ollama_source_is_carried_in_ssh_settings(monkeypatch):
    import json
    import shlex

    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    observed = []

    def run(command, **kwargs):
        # Windows CreateProcess supports at most 32767 command-line characters.
        assert len(subprocess.list2cmdline(command)) < 30000
        remote = shlex.split(command[-1])
        packed = json.loads(zlib.decompress(base64.b64decode(remote[-1])))
        observed.append(json.loads(packed["args"][0])["ollama"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", run)
    assert session["main"](["--user", "dangddk", "--allocate", "--start-ollama"]) == 0
    assert observed[0]["model"] == "qwen3.5:4b"
    assert observed[0]["port"] == 11435
    compile(observed[0]["source"], "compute_worker", "exec")


def test_connect_existing_job_uses_loopback_tunnel_and_no_allocation(monkeypatch):
    import shlex

    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"shell": False, "check": False}
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", run)
    assert (
        session["main"](
            [
                "--user",
                "dangddk",
                "--connect-job",
                "59660",
                "--local-port",
                "11440",
                "--relay-port",
                "11441",
                "--remote-port",
                "11442",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    command = calls[0]
    assert "ExitOnForwardFailure=yes" in command
    assert "127.0.0.1:11440:127.0.0.1:11441" in command
    remote = shlex.split(command[-1])
    packed = json.loads(zlib.decompress(base64.b64decode(remote[-1])))
    assert packed["args"] == [
        "--job-id",
        "59660",
        "--listen-port",
        "11441",
        "--upstream-port",
        "11442",
    ]
    assert "sbatch" not in packed["source"] and "scancel" not in packed["source"]
    compile(packed["source"], "relay", "exec")


@pytest.mark.parametrize(
    "options",
    [
        ["--connect-job", "0"],
        ["--connect-job", "-1"],
        ["--connect-job", "123", "--allocate"],
        ["--connect-job", "123", "--start-ollama"],
        ["--relay-port", "65536"],
    ],
)
def test_invalid_connection_settings_are_rejected(options):
    with pytest.raises(SystemExit):
        parse_args(["--user", "dangddk", *options])


def test_tunnel_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    monkeypatch.setattr(
        "subprocess.run", lambda command, **kw: subprocess.CompletedProcess(command, 255)
    )
    assert session["main"](["--user", "dangddk", "--connect-job", "123"]) == 255
    assert "Relay connection failed" in capsys.readouterr().err


def test_preserve_uses_existing_job_without_cancelling(monkeypatch):
    import shlex

    monkeypatch.setattr("shutil.which", lambda name: "ssh.exe")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        remote = shlex.split(command[-1])
        assert remote[0] == "srun" and "--jobid=59877" in remote
        assert "--gres=none" in remote
        payload = json.loads(zlib.decompress(base64.b64decode(remote[-1])))
        assert json.loads(payload["args"][0]) == {
            "action": "preserve",
            "job_id": 59877,
            "port": 11435,
        }
        assert kwargs["timeout"] == 180 and kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", run)
    assert session["main"](["--user", "dangddk", "--preserve-job", "59877"]) == 0
    assert len(calls) == 1


def test_remote_payload_roundtrip(monkeypatch):
    import shlex
    import sys

    remote = shlex.split(
        session["remote_python"]("captured = sys.argv[1:]", ["quote'", "line\nnext"])
    )
    namespace = {}
    monkeypatch.setattr(sys, "argv", ["-c", remote[-1]])
    exec(remote[3], namespace)
    assert namespace["captured"] == ["quote'", "line\nnext"]
