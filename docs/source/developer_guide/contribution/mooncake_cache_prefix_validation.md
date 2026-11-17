# Validate Mooncake cache prefixes on Atlas A2

This guide validates `AscendStoreConnector` with the Mooncake backend, including
bulk and layerwise transfers. It does not establish SSD reads or direct DMA.
Use a dedicated development environment and an isolated Mooncake master.

## Update both source checkouts

Set `WORK_ROOT` to the parent of your existing `vllm` and `vllm-ascend` checkouts.
Check for local changes before switching or updating branches. The following
commands do not discard local work; resolve a failed fast-forward explicitly.

```bash
export WORK_ROOT="$HOME/work/cache-prefix-arm64"
cd "$WORK_ROOT/vllm-ascend"
git status --short
git fetch origin codex/ascend-store-cache-prefix
git switch codex/ascend-store-cache-prefix
git merge --ff-only origin/codex/ascend-store-cache-prefix
git rev-parse HEAD
VLLM_COMMIT="$(tr -d '[:space:]' < .github/vllm-main-verified.commit)"
git -C "$WORK_ROOT/vllm" status --short
git -C "$WORK_ROOT/vllm" fetch --depth 1 origin "$VLLM_COMMIT"
git -C "$WORK_ROOT/vllm" checkout --detach FETCH_HEAD
```

For the initial clone, use the feature branch in your fork and initialize the
upstream vLLM checkout at `.github/vllm-main-verified.commit`. Do not substitute
an arbitrary vLLM release or the code bundled in a nightly image.

## Install for A2

Check `uname -m`, `npu-smi info`, and the driver/CANN compatibility first. A2
does not imply an ARM64 host. This branch's installation baseline is CANN
9.2.1, Python 3.12, torch 2.10.0 and torch-npu 2.10.0.post7; check the
[installation guide](../../getting_started/installation/install_vllm_ascend.inc.md)
and the checked-out requirements before changing an existing environment.

A virtual environment is optional. It isolates Python dependencies and editable
installs, but does not isolate the NPU, driver or CANN. In a dedicated development
container, use its Python directly and omit virtual-environment activation.
An ordinary virtual environment does not inherit system-installed Mooncake or
torch packages; install them in the selected environment and check import paths.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# Optional: create once, then activate on subsequent sessions.
python3.12 -m venv "$WORK_ROOT/venv"
source "$WORK_ROOT/venv/bin/activate"
python -m pip install --upgrade pip 'setuptools>=78.1.1,<81' setuptools-scm wheel
export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly'
cd "$WORK_ROOT/vllm-ascend"
python -m pip install -r requirements.txt
cd "$WORK_ROOT/vllm"
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e .
cd "$WORK_ROOT/vllm-ascend"
git submodule update --init --recursive
unset SOC_VERSION
python -m pip install --no-build-isolation -e .
python -m pip install --no-deps 'flash-linear-attention-npu-a2==26.9.1'
python -m pip install -r requirements-dev.txt -r requirements-lint.txt
python -m pip check
```

Let the build detect the actual A2 `ascend910b*` chip. If detection fails, set
`SOC_VERSION` from the hardware information; `ascend910b1` is an example, not a
universal A2 target. Do not reuse A3 build outputs or disable custom kernels for
NPU E2E. Install a validated Mooncake Ascend SDK and matching master: bulk needs
at least 0.3.11.post1; layerwise needs at least 0.3.14 and the session/range APIs.
Validate the available ARM64 wheel before selecting it; do not install a GPU SDK.

```bash
python - <<'PY'
import inspect
import torch
import torch_npu
import vllm
import vllm_ascend
from vllm_ascend._build_info import __device_type__
from mooncake.store import MooncakeDistributedStore
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MOONCAKE_LAYERWISE_CLIENT_METHODS
print(inspect.getfile(vllm), inspect.getfile(vllm_ascend))
print(torch.__version__, torch_npu.__version__, __device_type__)
assert __device_type__ == 'A2'
assert all(callable(getattr(MooncakeDistributedStore, name, None)) for name in MOONCAKE_LAYERWISE_CLIENT_METHODS)
print(torch.ones(1, device='npu'))
PY
```

## Run static checks and CPU tests first

On ARM64, preinstall native ShellCheck/Gitleaks and use Node.js >=20 with npm.
Verify both `node --version` and `npm --version`; node alone can still trigger
nodeenv's automatic download. Do not
assume an amd64 lint image or an x86-only automatic tool download will work.

```bash
cd "$WORK_ROOT/vllm-ascend"
export VIRTUALENV_NO_PERIODIC_UPDATE=1
set -o pipefail
bash format.sh ci 2>&1 | tee "$WORK_ROOT/format-ci.log"
export PYTHONPATH="$WORK_ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"
python .github/workflows/scripts/run_mypy_parallel.py 2>&1 | tee "$WORK_ROOT/mypy.log"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/distributed/ascend_store/test_cache_prefix.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_config.py \
  tests/ut/distributed/ascend_store/test_cache_prefix_probe.py \
  tests/ut/distributed/ascend_store/test_mooncake_pipeline.py \
  tests/ut/distributed/ascend_store/test_backend.py \
  tests/ut/distributed/ascend_store/test_layerwise_keys.py \
  tests/ut/distributed/ascend_store/test_metadata.py
```

The real-config tests must not skip in this environment. CPU mock results do
not establish upstream configuration compatibility or NPU correctness.

## Run two-instance NPU E2E

Use two idle A2 NPUs. Configure actual reachable IP/interface names on the
server; do not copy the placeholders literally. This is the A2 RoCE path.

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_IF_IP="<local-communication-IP>"
export GLOO_SOCKET_IFNAME="<interface>"
export TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export HCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export PYTHONHASHSEED=0
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_USE_MODELSCOPE=true python -m pytest -sv \
  tests/e2e/pull_request/two_card/test_mooncake_cache_prefix.py \
  -k 'literal and bulk' --basetemp="$WORK_ROOT/a2-prefix-smoke"
```

Use a new dedicated `--basetemp` directory: pytest deletes its contents. Remove
`-k` and select a different results directory for all six combinations of
bulk/layerwise and empty/literal/auto prefixes. They require 18 server starts.
The model is Qwen3-0.6B; prepare its cache before running.

The test keeps the writer alive, reads from a fresh reader with the same prefix,
then starts another reader with a different prefix or image identity. It checks
external lookup counters, actual GET/range-load evidence, zero local hits and
output agreement with the same-weight cold reference. It never retries a miss
on the same reader, since that reader could warm the pool itself. Inspect writer
PUT/session failures if the first same-prefix request misses.

Missing SDK/master or fewer than two visible NPUs cause explicit skips. An
installed SDK missing required layerwise APIs must fail startup. Directory
placement currently routes this test to A3 in PR CI; request A2 evidence
explicitly instead of treating an A3 job as A2 validation.

The reusable manual probe is
`tests/e2e/common/kv_pool/cache_prefix_probe.py --help`. Layerwise requires
`--mode layerwise --server-log <this-server-log>` and enabled
`{"kv_pool_range_debug": true}` additional configuration. Run it only against
idle test servers configured with `kv_load_failure_policy="fail"`.

## Additional acceptance and reporting

### Verify two services deployed by the operator

The probe also works as a standalone client. It starts no model, master or
container and changes no serving configuration. Keep both services alive;
the writer may contribute memory holding the objects being tested.

Both services can be PD-mixed nodes with `kv_role="kv_both"`: writer and reader
refer to request order, and both services can query, load and write the pool.
If choosing a read-only consumer instead, it should have
`consumer_is_to_load=true`, `consumer_is_to_put=false`, and
`save_decode_cache=false`. Disable local prefix caching and other test traffic.
For a hit test, use matching prefixes, compatible model weights/cache layouts,
and the same Mooncake master/tenant. For isolation, configure different prefixes
on the two services before running and expect a miss.

```bash
python -m pip install requests prometheus-client
python tests/e2e/common/kv_pool/cache_prefix_probe.py \
  --writer-url http://127.0.0.1:8100 \
  --reader-url http://127.0.0.1:8200 \
  --writer-model writer-served-name --reader-model reader-served-name \
  --expect hit --out ./prefix-hit.json
```

The `--*-model` values are served model names, not local weight paths. URL roots
and URLs ending in `/v1` are accepted. The default input uses `/tokenize` on both
servers, checks tokenization agreement, then builds 1536 token IDs shared by
both requests. A fresh run identifier prevents an earlier run from warming the
same prompt. No local tokenizer, torch, vLLM or NPU is needed by the client.

Use `--prompt-file <UTF-8-file>` for custom text of at least 512 tokens, or
`--payload <JSON-token-IDs-or-prior-report>` to bypass `/tokenize`. The latter must match both
models' token vocabulary. Change `--prompt-tokens` for the generated default
input, and `--max-tokens`/`--seed` for generation. Ensure prompt plus output fits
both services. `--model` supplies a shared served name instead of two names.

The client sends exactly one inference request to each service. It waits
`--wait-seconds` (default 5, maximum 60) after the writer; this is a grace
period, not a PUT completion fence. It never retries reader inference.
The writer is expected to miss by default; `--writer-expect hit` can validate
an explicitly prewarmed writer, but does not prove a new PUT occurred.

For different prefixes, reuse the successful hit report as input and require a
writer hit as a positive control:

```text
--payload ./prefix-hit.json --writer-expect hit --expect miss --out ./prefix-miss.json
```

The client rejects a paired miss test without `--writer-expect hit`: a reader
miss alone cannot distinguish isolation from missing writer objects. Keep the
writer and the first test's objects alive while reconfiguring the reader.
When intentionally using different model weights, add `--skip-output-compare`;
each model's accuracy still needs its own cold-reference validation.
For authentication, `--api-key-env <variable-name>` reads a shared key from the
client environment without saving it in the report.

Layerwise verification additionally needs `--mode layerwise --writer-log
<writer-log> --reader-log <reader-log>` and enabled range-debug logging on both
services. The client must be able to read the live log files, for example by
running on the A2 host. Bulk mode can run remotely using HTTP alone.
For complete `kv_both` configuration examples and copyable layerwise hit/miss
commands, see the [implementation and validation notes](../Design_Documents/mooncake_cache_prefix.md#55-两端-kv_both-的-layerwise-验证).
The client is committed at
[`tests/e2e/common/kv_pool/cache_prefix_probe.py`](../../../../tests/e2e/common/kv_pool/cache_prefix_probe.py).
A `kv_both` reader can publish after its first miss; a repeat isolation test
needs a reader namespace without that input. Restarting the reader alone does
not clear objects already stored in Mooncake.

The console shows both generated texts, request durations, metric deltas and
the verdict. The JSON report also saves token IDs, model names, full responses,
and the stage/error on failure. Exit status 0 means validation passed; 1 means
validation/request failure; 2 means invalid CLI arguments. Expected hit requires
lookup and load evidence; expected miss rejects external loading. It cannot
inspect or change `cache_prefix`, and a hit alone does not prove physical SSD I/O.

### Reset and model-specific acceptance

For bulk reset, populate two prefixes, stop requests and drain all writers,
then request connector reset through the server's available administration API.
Both prefixes must miss afterwards. Do not use a local-only cache reset or a
production master. Layerwise external reset remains unsupported.

Validate DSV4.1 + DSpark separately on its existing supported topology. Set the
prefix on the AscendStoreConnector child of MultiConnector; retain target/draft
models and TP/DP/PP settings. Verify same-prefix reuse, different-prefix misses,
real draft execution and correct output. This does not enable Memcache-specific
GVA companion objects on Mooncake.

Conversion follows upstream: JSON null/false/0 become literal `"None"`,
`"False"`, `"0"`. Explicit strings are recommended. Nonempty values on Memcache
or Yuanrong fail early; omitted/empty values preserve their existing behavior.

Record repository SHAs, package paths and versions, CANN/driver/NPU, model,
commands, logs, response JSON, skips and failures. Report Windows CPU mocks,
real-config CPU tests, format/mypy and A2 E2E as separate evidence.
