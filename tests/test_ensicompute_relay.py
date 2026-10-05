import base64
import json
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

relay = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/ensicompute_relay.py"))
)


class RelayTests(unittest.TestCase):
    def test_forward_uses_existing_job_without_shell(self):
        upstream = b'{"version":"0.40.0"}'
        reply = json.dumps({"status": 200, "body": base64.b64encode(upstream).decode()})
        with patch.object(
            relay.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=reply)
        ) as run:
            self.assertEqual(
                relay.forward(59623, 11435, "GET", "/api/version", b""), (200, upstream)
            )
            command = run.call_args.args[0]
            self.assertIn("--jobid=59623", command)
            self.assertIn("--gres=none", command)
            self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_stream_is_disabled_when_omitted(self):
        reply = json.dumps({"status": 200, "body": "e30="})
        with patch.object(
            relay.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=reply)
        ) as run:
            relay.forward(59623, 11435, "POST", "/api/chat", b'{"model":"qwen3.5:4b"}')
            packet = json.loads(run.call_args.kwargs["input"])
            self.assertIs(json.loads(base64.b64decode(packet["body"]))["stream"], False)

    def test_unexpected_requests_do_not_launch_steps(self):
        with patch.object(relay.subprocess, "run") as run:
            for method, path, body in [
                ("POST", "/api/pull", b"{}"),
                ("POST", "/api/chat", b'{"stream":true}'),
                ("POST", "/api/chat", b"[]"),
                ("GET", "/api/version", b"x" * (relay.LIMIT + 1)),
            ]:
                with self.assertRaises(ValueError):
                    relay.forward(59623, 11435, method, path, body)
            run.assert_not_called()

    def test_failed_slurm_job_is_reported(self):
        with patch.object(
            relay.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1, stderr="Job is no longer running"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Slurm request failed"):
                relay.forward(59623, 11435, "GET", "/api/version", b"")


if __name__ == "__main__":
    unittest.main()
