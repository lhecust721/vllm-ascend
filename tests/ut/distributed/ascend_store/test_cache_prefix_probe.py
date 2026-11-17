# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.e2e.common.kv_pool import cache_prefix_probe as probe
from tests.e2e.common.kv_pool.cache_prefix_probe import completed_range_bytes, parse_metrics, validate_observation


class TestCachePrefixProbe(unittest.TestCase):
    def test_missing_metrics_are_not_interpreted_as_zero_hits(self):
        with self.assertRaisesRegex(AssertionError, "Missing prefix metrics"):
            parse_metrics("vllm:external_prefix_cache_queries_total 10\n")
        metrics = parse_metrics(
            "\n".join(
                [
                    'vllm:external_prefix_cache_queries_total{engine="0"} 10',
                    'vllm:external_prefix_cache_hits_total{engine="0"} 8',
                    'vllm:prefix_cache_hits_total{engine="0"} 0',
                ]
            )
        )
        self.assertEqual(metrics, {"external_queries": 10, "external_hits": 8, "local_hits": 0})

    def test_range_evidence_excludes_saves_and_rejects_short_or_failed_gets(self):
        event = {"event": "range", "direction": "load", "requested_bytes": [16, 32], "results": [16, 32]}
        line = "worker [KVPOOL_RANGE_DEBUG] " + json.dumps(event)
        self.assertEqual(completed_range_bytes(line), 48)
        self.assertEqual(completed_range_bytes(line.replace('"load"', '"save"')), 0)
        for results in ([16], [-1, 32], [0, 32], [8, 32]):
            event["results"] = results
            with self.assertRaises(AssertionError):
                completed_range_bytes("[KVPOOL_RANGE_DEBUG] " + json.dumps(event))

    def test_lookup_hit_requires_load_and_miss_rejects_loading(self):
        for mode in ("bulk", "layerwise"):
            record: dict = {
                "mode": mode,
                "metrics_delta": {"external_queries": 512, "external_hits": 384, "local_hits": 0, "bulk_get_keys": 0},
                "range_load_bytes": 0,
            }
            with self.assertRaisesRegex(AssertionError, "completed load"):
                validate_observation(record, "hit")
            if mode == "bulk":
                record["metrics_delta"]["bulk_get_keys"] = 3
            else:
                record["range_load_bytes"] = 1024
            validate_observation(record, "hit")
            record["metrics_delta"]["external_hits"] = 0
            with self.assertRaisesRegex(AssertionError, "unexpectedly restored"):
                validate_observation(record, "miss")

    def test_capture_ignores_old_log_records_and_accepts_only_new_loads(self):
        event = {"event": "range", "direction": "load", "requested_bytes": [16], "results": [16]}
        line = "[KVPOOL_RANGE_DEBUG] " + json.dumps(event) + "\n"
        before = {"external_queries": 0, "external_hits": 0, "local_hits": 0}
        after = {**before, "external_queries": 512, "external_hits": 384}
        for append_new_load in (False, True):
            with self.subTest(append_new_load=append_new_load), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "server.log"
                log.write_text(line, encoding="utf-8")

                def respond(*args, append_new_load=append_new_load, log=log, **kwargs):
                    if append_new_load:
                        with log.open("a", encoding="utf-8") as output:
                            output.write(line)
                    response = MagicMock()
                    response.json.return_value = {"choices": [{"text": "expected"}]}
                    return response

                with (
                    patch.object(probe, "snapshot", side_effect=[before, after]),
                    patch.object(probe.requests, "post", side_effect=respond) as post,
                    patch.object(probe.time, "monotonic", side_effect=[0, 1, 2, 1000]),
                ):
                    record = probe.capture_request("http://test", [1] * 512, mode="layerwise", log_path=log)
                post.assert_called_once()
                self.assertEqual(record["range_load_bytes"], 16 if append_new_load else 0)
                if append_new_load:
                    validate_observation(record, "hit")
                else:
                    with self.assertRaisesRegex(AssertionError, "completed load"):
                        validate_observation(record, "hit")
