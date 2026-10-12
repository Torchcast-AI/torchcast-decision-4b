# Torchcast Decision 4B (v1.1)

A text-based decision model developed by **Torchcast AI**, based on Qwen3.5-4B. It returns yes/no probabilities
(`noul`), choice distributions (`choice`) and ordinal score distributions (`score`) through a TypeSafe-compatible API.

[Model card and weights](https://huggingface.co/torchcast-ai/torchcast-decision-4b/tree/v1.1.0)

## Run for evaluation

Linux with one NVIDIA GPU (tested on one A40 48 GB, BF16). The weights repository contains the serving package
(`torchcast_decision/`) and its configuration; use both unchanged.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.8.1" "flash-linear-attention==0.5.2" accelerate safetensors "huggingface_hub[cli]" \
    ninja packaging wheel setuptools
pip install --no-build-isolation "causal-conv1d>=1.7"

hf download torchcast-ai/torchcast-decision-4b --revision v1.1.0 --local-dir torchcast-decision-4b
cd torchcast-decision-4b
sha256sum -c SHA256SUMS
python -m torchcast_decision.check .                 # fast kernels must be active
python -m torchcast_decision.server --model . --host 127.0.0.1 --port 8011
```

The pinned artifact is `torchcast-ai/torchcast-decision-4b@f4a34e7aedbf22917f6c85b8c4e03a3da49c1baf` (tag `v1.1.0`);
weight SHA-256 values are in its `SHA256SUMS`.

Check the served model name, then send a warm-up request before timing:

```bash
curl -s http://127.0.0.1:8011/v1/models        # "name": "torchcast-decision-4b-v1.1"
curl --fail http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer reports two charges for one order.","questions":{"decision":{"type":"choice","instructions":"Choose the support team.","criteria":{"billing":"Payments and refunds","technical":"Product malfunctions","other":"Other requests"}}}}'
```

For JevBench, use the `typesafe` adapter against `http://127.0.0.1:8011` with model name
`torchcast-decision-4b-v1.1`, for example from a JevBench checkout at commit
`7625ecb1b075cfe4d09d34141d353325825eea1d`:

```bash
J=datasets/public
python -m jevbench.cli run \
  --tasks $J/easy.jsonl,$J/original.jsonl,$J/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8011 --key-env "" \
  --model torchcast-decision-4b-v1.1 --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results runs/results.jsonl --raw-dir runs/raw --ledger runs/ledger.jsonl \
  --manifest runs/manifest.json --delay-s 0
```

The bundled `decision_config.json` applies a calibration temperature of 0.9 and a yes/no commit floor of 0.801
(the answer does not change); prompts longer than 16,384 tokens keep the head and the tail of the state, as
`/health` reports. See the weights repository's `NOTICE`.

The server listens on 127.0.0.1 by default; if the benchmark client runs on another machine, pass `--host 0.0.0.0`
and restrict access at the network level. This checkpoint has no published provider tariff; apply the benchmark's
documented estimated-cost method.

## Numerical reproducibility

Measured with this release on one NVIDIA A40 48 GB (driver 570.195.03, CUDA 12.8, torch 2.9.1, transformers 5.8.1,
flash-linear-attention 0.5.2, Triton 3.5.1, BF16), all 231 JevBench public items, one question per request:

| Condition | Max probability difference | Predicted labels |
|---|---|---|
| Same server: repeated, back-to-back, shuffled order | 0 (bit-identical) | identical |
| Server restart, warm Triton cache | 0 (bit-identical) | identical |
| Fresh Triton cache (cold) vs another fresh cache vs warm | 0 (bit-identical) | identical |

The linear-attention layers run Triton kernels that are normally autotuned: on a fresh cache each kernel times its
candidate block configurations and keeps the fastest. Timing noise can select a different configuration, which changes
the BF16 summation order; without pinning we measured up to 0.030 difference between cache states (11 to 25 of the 231
items above 0.005, no predicted label changed). The serving package therefore pins the selection
(`torchcast_decision/_pin_triton.py`): every candidate configuration is compiled and run once, so configurations that
do not fit the GPU are skipped, and the first valid configuration in the kernel's own list is used; no timings are
read from or written to the on-disk cache. Latency is unchanged (p50 32 ms, p95 356 ms on the A40 for these items).
Set `TC_TRITON_AUTOTUNE=1` to restore timing-based selection.

Different GPU models or software versions use different kernels and libraries (for example cuBLAS algorithms and
the valid Triton configurations) and can still give slightly different probabilities for the same request; compare
runs on the same GPU type and software.

## Licence

Model weights: CC BY-NC 4.0 (non-commercial). Base model: Qwen3.5-4B, Apache License 2.0, Copyright 2026 Alibaba Cloud. Serving package: Apache
License 2.0. See the licence files in the weights repository. This project is independent of TypeSafe AI and the
JevBench maintainers.
