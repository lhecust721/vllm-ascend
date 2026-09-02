"""Run a two-turn Qwen3 Decode-to-KV-Pool reuse check.

Before running, replace ``<API或路由节点>`` with the deployed endpoint.
The server should be started with ``qwen3_multiturn_kv_reuse.jinja``.
"""

import json
import time
import uuid

import requests


URL = "http://<API或路由节点>:8000/v1/chat/completions"
MODEL = "Qwen3-1.7B"

verify_id = str(uuid.uuid4())

messages1 = [
    {
        "role": "system",
        "content": f"KV验证ID={verify_id}。请严格按照用户要求回答。",
    },
    {
        "role": "user",
        "content": "请连续介绍人工智能的发展历史，输出至少300个token，不要提前结束。",
    },
]

payload1 = {
    "model": MODEL,
    "messages": messages1,
    "temperature": 0,
    "max_tokens": 320,
    "min_tokens": 260,
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}

r1 = requests.post(URL, json=payload1, timeout=600)
r1.raise_for_status()
data1 = r1.json()

assistant1 = data1["choices"][0]["message"]["content"]
usage1 = data1["usage"]

print("turn1 request id:", data1.get("id"))
print("turn1 prompt tokens:", usage1["prompt_tokens"])
print("turn1 completion tokens:", usage1["completion_tokens"])

messages2 = messages1 + [
    {
        "role": "assistant",
        "content": assistant1,
    },
    {
        "role": "user",
        "content": "请用一句话总结前面的回答。",
    },
]

payload2 = {
    "model": MODEL,
    "messages": messages2,
    "temperature": 0,
    "max_tokens": 32,
    "stream": True,
    "stream_options": {
        "include_usage": True,
    },
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}

start = time.perf_counter()
ttft = None
usage2 = None

with requests.post(URL, json=payload2, stream=True, timeout=600) as r2:
    r2.raise_for_status()

    for line in r2.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        payload = line[6:]
        if payload == "[DONE]":
            break

        event = json.loads(payload)

        if event.get("usage"):
            usage2 = event["usage"]

        choices = event.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if delta.get("content") and ttft is None:
                ttft = time.perf_counter() - start

print("turn2 TTFT:", ttft)
print("turn2 usage:", usage2)
