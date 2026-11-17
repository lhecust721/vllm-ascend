# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe idle test servers; prove namespace lookup and load, not SSD/DMA."""

import argparse
import json
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families

METRICS = {
    "external_queries": "vllm:external_prefix_cache_queries",
    "external_hits": "vllm:external_prefix_cache_hits",
    "local_hits": "vllm:prefix_cache_hits",
    "bulk_get_keys": "vllm:ascend_store_load_get_keys",
}
REQUIRED_METRICS = {"external_queries", "external_hits", "local_hits"}
RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"
METRICS_TIMEOUT_SECONDS = 60
HTTP_TIMEOUT_SECONDS = 600
MIN_PROMPT_TOKENS = 512


def parse_metrics(payload: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            for label, name in METRICS.items():
                if sample.name in (name, name + "_total"):
                    values[label] = values.get(label, 0.0) + sample.value
    missing = REQUIRED_METRICS - values.keys()
    if missing:
        raise AssertionError(f"Missing prefix metrics: {sorted(missing)}")
    return values


def snapshot(url: str) -> dict[str, float]:
    response = requests.get(url.rstrip("/") + "/metrics", timeout=30)
    response.raise_for_status()
    return parse_metrics(response.text)


def completed_range_bytes(text: str) -> int:
    """Count successful load bytes; reject partial/failed SDK range results."""
    loaded = 0
    for line in text.splitlines():
        if RANGE_DEBUG_PREFIX not in line:
            continue
        payload = line.split(RANGE_DEBUG_PREFIX, 1)[1].lstrip()
        event, _ = json.JSONDecoder().raw_decode(payload)
        if event.get("event") != "range" or event.get("direction") != "load":
            continue
        requested, results = event["requested_bytes"], event["results"]
        if not requested or len(requested) != len(results):
            raise AssertionError(f"Invalid range result: {event}")
        if any(size <= 0 or result != size for size, result in zip(requested, results, strict=True)):
            raise AssertionError(f"Range GET did not restore the requested bytes: {event}")
        loaded += sum(requested)
    return loaded


def capture_request(
    url: str,
    tokens: list[int],
    *,
    model: str = "kv-prefix-e2e",
    mode: str = "bulk",
    log_path: Path | None = None,
    expect: str = "hit",
) -> dict:
    if (
        not isinstance(tokens, list)
        or len(tokens) < MIN_PROMPT_TOKENS
        or any(type(token) is not int for token in tokens)
    ):
        raise ValueError(f"payload must contain at least {MIN_PROMPT_TOKENS} integer token IDs")
    if mode not in ("bulk", "layerwise") or expect not in ("hit", "miss"):
        raise ValueError("mode must be bulk/layerwise and expect must be hit/miss")
    if mode == "layerwise" and log_path is None:
        raise ValueError("layerwise mode needs the server log to verify completed range GETs")
    offset = log_path.stat().st_size if log_path is not None else 0
    before = snapshot(url)
    started = time.monotonic()
    response = requests.post(
        url.rstrip("/") + "/v1/completions",
        json={
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 16,
            "ignore_eos": True,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    elapsed = time.monotonic() - started
    deadline = time.monotonic() + METRICS_TIMEOUT_SECONDS
    while True:
        after = snapshot(url)
        delta = {name: after.get(name, 0) - before.get(name, 0) for name in METRICS}
        loaded = 0
        if log_path is not None:
            with log_path.open("rb") as log:
                log.seek(offset)
                loaded = completed_range_bytes(log.read().decode("utf-8", errors="replace"))
        load_ready = expect == "miss" or (delta["bulk_get_keys"] > 0 if mode == "bulk" else loaded > 0)
        if (delta["external_queries"] > 0 and load_ready) or time.monotonic() >= deadline:
            break
        time.sleep(1)
    return {
        "url": url,
        "mode": mode,
        "elapsed_seconds": elapsed,
        "metrics_delta": delta,
        "range_load_bytes": loaded,
        "response": result,
    }


def validate_observation(record: dict, expect: str, reference: dict | None = None) -> None:
    delta = record["metrics_delta"]
    if any(value < 0 for value in delta.values()):
        raise AssertionError(f"Counters reset during the request: {delta}")
    if delta["external_queries"] <= 0:
        raise AssertionError("No external query observed; this run cannot establish a Store hit/miss")
    if delta["local_hits"] != 0:
        raise AssertionError("Local prefix hits confound the result; disable local caching and other traffic")
    if (delta["external_hits"] > 0) != (expect == "hit"):
        raise AssertionError(f"Expected {expect}, got {delta}")
    loaded = delta["bulk_get_keys"] if record["mode"] == "bulk" else record["range_load_bytes"]
    if expect == "hit" and loaded <= 0:
        raise AssertionError("Lookup hit without completed load evidence")
    if expect == "miss" and loaded > 0:
        raise AssertionError("A namespace miss unexpectedly restored external KV")
    if (
        reference is not None
        and record["response"]["choices"][0]["text"] != reference["response"]["choices"][0]["text"]
    ):
        raise AssertionError("Generated text differs from the same-weight cold reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="kv-prefix-e2e")
    parser.add_argument("--expect", required=True, choices=("hit", "miss"))
    parser.add_argument("--mode", choices=("bulk", "layerwise"), default="bulk")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    record = capture_request(
        args.url,
        json.loads(args.payload.read_text(encoding="utf-8")),
        model=args.model,
        mode=args.mode,
        log_path=args.server_log,
        expect=args.expect,
    )
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
        validate_observation(record, args.expect, reference)
        record["status"] = "passed"
    except AssertionError as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps({"output": str(args.out), "status": record["status"], "metrics_delta": record["metrics_delta"]})
        )


if __name__ == "__main__":
    main()
