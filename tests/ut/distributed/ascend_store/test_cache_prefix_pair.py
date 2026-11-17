# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP/CLI control-flow tests, not a substitute for real Mooncake/NPU E2E."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.e2e.common.kv_pool import cache_prefix_probe as probe


@contextmanager
def fake_service(name, namespace, objects, calls):
    counters = {label: 0 for label in probe.METRICS}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, content_type="application/json"):
            content = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            assert self.path == "/metrics"
            self.respond(
                "\n".join(f"{metric}_total {counters[label]}" for label, metric in probe.METRICS.items()), "text/plain"
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.respond({"tokens": list(body["prompt"].encode()), "max_model_len": 4096})
                return
            assert self.path == "/v1/completions"
            calls.append((name, body))
            key = (namespace, tuple(body["prompt"]))
            counters["external_queries"] += len(body["prompt"])
            if key in objects:
                counters["external_hits"] += len(body["prompt"]) - 128
                counters["bulk_get_keys"] += 1
            if name == "writer":
                objects.add(key)
            self.respond({"choices": [{"text": "same-weight deterministic output"}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestPairedCachePrefixCLI(unittest.TestCase):
    def run_pair(self, namespace, expect):
        objects: set[tuple[str, tuple[int, ...]]] = set()
        calls: list[tuple[str, dict]] = []
        with ExitStack() as stack:
            writer = stack.enter_context(fake_service("writer", "v1", objects, calls))
            reader = stack.enter_context(fake_service("reader", namespace, objects, calls))
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            out = Path(directory) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    probe.__file__,
                    "--writer-url",
                    writer + "/v1",
                    "--reader-url",
                    reader,
                    "--writer-model",
                    "writer-model",
                    "--reader-model",
                    "reader-model",
                    "--expect",
                    expect,
                    "--wait-seconds",
                    "0",
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([name for name, _ in calls], ["writer", "reader"])
        self.assertEqual(calls[0][1]["prompt"], calls[1][1]["prompt"])
        self.assertEqual(len(calls[0][1]["prompt"]), 1536)
        self.assertEqual(calls[0][1]["model"], "writer-model")
        self.assertEqual(calls[1][1]["model"], "reader-model")
        self.assertEqual(report["writer"]["metrics_delta"]["external_hits"], 0)
        self.assertIn("same-weight deterministic output", result.stdout)
        return result, report

    def test_same_prefix_hit_and_different_prefix_miss(self):
        for namespace, expect in (("v1", "hit"), ("v2", "miss")):
            with self.subTest(namespace=namespace):
                result, report = self.run_pair(namespace, expect)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["stage"], "complete")

    def test_unexpected_reader_hit_is_reported_and_exits_nonzero(self):
        result, report = self.run_pair("v1", "miss")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "reader")
        self.assertIn("Expected miss", report["error"])

    def test_payload_file_bypasses_tokenization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.json"
            path.write_text(json.dumps([1] * 512), encoding="utf-8")
            args = type("Args", (), {"payload": path})()
            self.assertEqual(probe.pair_tokens(args, "writer", "reader", {}), [1] * 512)
