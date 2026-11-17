# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe idle test servers; prove namespace lookup and load, not SSD/DMA."""

import argparse
import hashlib
import json
import os
import time
import uuid
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


def server_root(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


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


def snapshot(url: str, headers: dict[str, str] | None = None) -> dict[str, float]:
    response = requests.get(server_root(url) + "/metrics", headers=headers, timeout=30)
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
    max_tokens: int = 16,
    seed: int = 42,
    headers: dict[str, str] | None = None,
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
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    offset = log_path.stat().st_size if log_path is not None else 0
    before = snapshot(url, headers)
    started = time.monotonic()
    response = requests.post(
        server_root(url) + "/v1/completions",
        json={
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
        headers=headers,
    )
    response.raise_for_status()
    result = response.json()
    choices = result.get("choices") if isinstance(result, dict) else None
    if (
        not choices
        or not isinstance(choices, list)
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("text"), str)
    ):
        raise ValueError("/v1/completions did not return a generated text choice")
    elapsed = time.monotonic() - started
    deadline = time.monotonic() + METRICS_TIMEOUT_SECONDS
    while True:
        after = snapshot(url, headers)
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


def tokenize(url: str, model: str, text: str, headers: dict[str, str] | None = None) -> list[int]:
    response = requests.post(
        server_root(url) + "/tokenize",
        json={
            "model": model,
            "prompt": text,
            "add_special_tokens": False,
        },
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("/tokenize did not return a nonempty integer token list; use --payload if unavailable")
    return tokens


def pair_tokens(args, writer_model: str, reader_model: str, headers: dict[str, str]) -> list[int]:
    if args.payload:
        return json.loads(args.payload.read_text(encoding="utf-8"))
    custom = args.prompt is not None or args.prompt_file is not None
    text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not custom:
        # A new first hash block prevents an earlier run from warming this
        # run's prefix, without modifying either server's deployment config.
        text = f"Namespace verification run {uuid.uuid4().hex}. Explain why reliable cache isolation matters. "
    assert text is not None
    writer_tokens = tokenize(args.writer_url, writer_model, text, headers)
    reader_tokens = tokenize(args.reader_url, reader_model, text, headers)
    if writer_tokens != reader_tokens:
        raise AssertionError("Writer and reader tokenize the input differently; verify their model/tokenizer pairing")
    if custom:
        return writer_tokens
    return (writer_tokens * (args.prompt_tokens // len(writer_tokens) + 1))[: args.prompt_tokens]


def verify_pair(args) -> tuple[dict, int]:
    """One writer request followed by one fresh-reader request; never deploy."""
    writer_model, reader_model = args.writer_model or args.model, args.reader_model or args.model
    record: dict = {
        "status": "failed",
        "stage": "input",
        "writer_model": writer_model,
        "reader_model": reader_model,
        "writer_expected": args.writer_expect,
        "reader_expected": args.expect,
        "output_comparison": not args.skip_output_compare,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "wait_seconds": args.wait_seconds,
    }
    try:
        headers = {}
        if args.api_key_env:
            key = os.environ.get(args.api_key_env)
            if not key:
                raise ValueError(f"API key environment variable {args.api_key_env!r} is empty")
            headers = {"Authorization": f"Bearer {key}"}
        tokens = pair_tokens(args, writer_model, reader_model, headers)
        record["prompt_token_ids"] = tokens
        record["prompt_tokens"] = len(tokens)
        record["prompt_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        record["stage"] = "writer"
        print("Sending writer request...", flush=True)
        record["writer"] = capture_request(
            args.writer_url,
            tokens,
            model=writer_model,
            mode=args.mode,
            log_path=args.writer_log,
            expect=args.writer_expect,
            max_tokens=args.max_tokens,
            seed=args.seed,
            headers=headers,
        )
        validate_observation(record["writer"], args.writer_expect)
        record["stage"] = "settle"
        # This grace period is not a PUT completion fence. A missing/failed
        # writer PUT must still fail the first reader request, never be hidden
        # by retries that could warm a write-enabled reader's own objects.
        print(f"Waiting {args.wait_seconds:g}s for writer PUTs; keeping writer alive...", flush=True)
        time.sleep(args.wait_seconds)
        record["stage"] = "reader"
        print("Sending reader request (one attempt)...", flush=True)
        record["reader"] = capture_request(
            args.reader_url,
            tokens,
            model=reader_model,
            mode=args.mode,
            log_path=args.reader_log,
            expect=args.expect,
            max_tokens=args.max_tokens,
            seed=args.seed,
            headers=headers,
        )
        reference = None if args.skip_output_compare else record["writer"]
        validate_observation(record["reader"], args.expect, reference)
        record.update(status="passed", stage="complete")
        return record, 0
    except (AssertionError, ValueError, KeyError, OSError, requests.RequestException) as error:
        record["error"] = str(error)
        return record, 1


def pair_summary(record: dict) -> dict:
    summary = {name: record[name] for name in ("status", "stage", "reader_expected")}
    summary["error"] = record.get("error")
    summary["prompt_tokens"] = record.get("prompt_tokens")
    for name in ("writer", "reader"):
        if name in record:
            observation = record[name]
            choices = observation.get("response", {}).get("choices", [])
            summary[name] = {
                "generated_text": choices[0].get("text") if choices else None,
                "elapsed_seconds": observation["elapsed_seconds"],
                "metrics_delta": observation["metrics_delta"],
                "range_load_bytes": observation["range_load_bytes"],
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--url", help="single-server probe; accepts root URL or /v1")
    endpoint.add_argument("--writer-url", help="paired test: already deployed writer root URL or /v1")
    parser.add_argument("--reader-url", help="paired test: already deployed fresh/read-only reader")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--payload", type=Path, help="JSON integer token IDs; bypass /tokenize")
    inputs.add_argument("--prompt-file", type=Path, help="UTF-8 text, at least 512 tokens; paired test only")
    inputs.add_argument("--prompt", help="text, at least 512 tokens; paired test only")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="kv-prefix-e2e")
    parser.add_argument("--expect", default="hit", choices=("hit", "miss"), help="expected reader/single-server result")
    parser.add_argument("--writer-expect", default="miss", choices=("hit", "miss"))
    parser.add_argument("--writer-model", help="writer served-model-name; defaults to --model")
    parser.add_argument("--reader-model", help="reader served-model-name; defaults to --model")
    parser.add_argument("--prompt-tokens", type=int, default=1536, help="length of generated default input")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-seconds", type=float, default=5, help="0..60s grace period; not a PUT completion fence")
    parser.add_argument("--skip-output-compare", action="store_true", help="for intentionally different model weights")
    parser.add_argument("--api-key-env", help="environment variable holding a shared API key; paired mode only")
    parser.add_argument("--mode", choices=("bulk", "layerwise"), default="bulk")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--writer-log", type=Path, help="writer log accessible on this machine; required for layerwise")
    parser.add_argument("--reader-log", type=Path, help="reader log accessible on this machine; required for layerwise")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.writer_url:
        if not args.reader_url or server_root(args.writer_url) == server_root(args.reader_url):
            parser.error("paired mode requires two distinct --writer-url and --reader-url endpoints")
        if not 0 <= args.wait_seconds <= 60 or args.prompt_tokens < MIN_PROMPT_TOKENS or args.max_tokens <= 0:
            parser.error("require 0 <= --wait-seconds <= 60, --prompt-tokens >= 512 and --max-tokens > 0")
        if args.mode == "layerwise" and (args.writer_log is None or args.reader_log is None):
            parser.error("layerwise paired mode requires --writer-log and --reader-log")
        record, code = verify_pair(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(pair_summary(record), ensure_ascii=False, indent=2))
        print(f"Full report: {args.out}")
        raise SystemExit(code)
    if args.payload is None:
        parser.error("single-server mode requires --payload; use --writer-url/--reader-url for generated input")
    record = capture_request(
        args.url,
        json.loads(args.payload.read_text(encoding="utf-8")),
        model=args.model,
        mode=args.mode,
        log_path=args.server_log,
        expect=args.expect,
        max_tokens=args.max_tokens,
        seed=args.seed,
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
