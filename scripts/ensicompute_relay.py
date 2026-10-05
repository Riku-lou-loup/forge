"""Session-only Ollama relay through an existing, user-owned Slurm allocation.

Run on the login node. Only the small HTTP relay runs there; each upstream
request executes inside the specified allocation, where Ollama is running.
No additional allocation, SSH key, package, or model download is required.
"""

import argparse
import base64
import http.server
import json
import signal
import subprocess
import sys

LIMIT = 1_048_576
ROUTES = {
    "GET": {"/api/version", "/api/tags", "/api/ps"},
    "POST": {"/api/show", "/api/chat", "/api/generate"},
}

# No shell interpolation. The helper receives the request through stdin.
WORKER = r"""
import base64, http.client, json, sys
request = json.load(sys.stdin)
connection = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=180)
try:
    connection.request(request["method"], request["path"],
        base64.b64decode(request["body"]), {"Content-Type": "application/json"})
    response = connection.getresponse()
    body = response.read(1048577)
    if len(body) > 1048576:
        raise ValueError("Upstream response exceeds the relay limit")
    print(json.dumps({"status": response.status,
        "body": base64.b64encode(body).decode("ascii")}))
finally:
    connection.close()
"""


def forward(job_id, upstream_port, method, path, body):
    if path not in ROUTES.get(method, set()):
        raise ValueError("Unsupported Ollama endpoint")
    if len(body) > LIMIT:
        raise ValueError("Request exceeds the relay limit")
    if method == "POST":
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        if payload.get("stream") is True:
            raise ValueError("Use stream:false through this relay")
        if path in {"/api/chat", "/api/generate"}:
            payload["stream"] = False
        body = json.dumps(payload).encode("utf-8")
    command = [
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
        WORKER,
        str(upstream_port),
    ]
    packet = {"method": method, "path": path, "body": base64.b64encode(body).decode("ascii")}
    result = subprocess.run(
        command, input=json.dumps(packet), capture_output=True, text=True, timeout=210, check=False
    )
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr, flush=True)
        raise RuntimeError("Slurm request failed. Check that the job and Ollama are still running.")
    reply = json.loads(result.stdout)
    status = reply["status"]
    content = base64.b64decode(reply["body"], validate=True)
    if type(status) is not int or not 100 <= status <= 599 or len(content) > LIMIT:
        raise ValueError("Invalid upstream response")
    return status, content


class Handler(http.server.BaseHTTPRequestHandler):
    def handle_request(self):
        self.connection.settimeout(20)
        try:
            if self.path not in ROUTES.get(self.command, set()):
                self.send_error(404, "Endpoint not supported by this relay")
                return
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked requests are not supported")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= LIMIT:
                raise ValueError("Invalid request size")
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Incomplete request body")
            status, content = forward(
                self.server.job_id, self.server.upstream_port, self.command, self.path, body
            )
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            status = 502
            content = json.dumps({"error": str(error)}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(content)

    do_GET = handle_request
    do_POST = handle_request


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--listen-port", type=int, default=11436)
    parser.add_argument("--upstream-port", type=int, default=11435)
    args = parser.parse_args()
    if args.job_id <= 0 or not all(
        1024 <= p <= 65535 for p in (args.listen_port, args.upstream_port)
    ):
        parser.error("Use a positive job ID and ports between 1024 and 65535")
    # One request at a time: a demo needs no concurrent model generations.
    with http.server.HTTPServer(("127.0.0.1", args.listen_port), Handler) as server:
        server.job_id = args.job_id
        server.upstream_port = args.upstream_port
        status, content = forward(args.job_id, args.upstream_port, "GET", "/api/version", b"")
        version = json.loads(content)
        if status != 200 or not isinstance(version, dict) or not version.get("version"):
            raise RuntimeError("The selected job did not return an Ollama version")
        print(f"Verified Ollama {version['version']} through job {args.job_id}.", flush=True)
        print(f"Relay ready on 127.0.0.1:{args.listen_port}; Slurm job {args.job_id}.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Relay failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
