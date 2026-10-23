# Torchcast Decision 4B (v1.3)

A text-based decision model developed by **Torchcast AI**, based on Qwen3.5-4B. It returns yes/no probabilities
(`noul`), choice distributions (`choice`) and ordinal score distributions (`score`) through a TypeSafe-compatible API.

[Model card and weights](https://huggingface.co/torchcast-ai/torchcast-decision-4b/tree/v1.3.0)

## What changed in v1.3

- **Weights:** a new full fine-tune of `Qwen/Qwen3.5-4B` (revision `851bf6e8`, same recipe as v1.2, 3,505 steps) on a
  new training data mix of 897,139 rows: the earlier v1.3 build's 878,791 rows plus 18,348 new rows from a further
  non-Claude data block, with 29,385 question-bank rows reworded by MiniMax, re-smoothed targets for easy and standard
  items (label smoothing 0.1 to 0.03 on 267,530 rows) and 18,877 paired twin items placed in the same batches. The
  architecture, tokenizer, prompt format (`jev_render_v3`) and non-causal attention mode are the same as in v1.2.
- **Speed:** the non-causal mode now runs with CUDA graphs (`torchcast_decision/model.py`). Each graph records a
  key-padding mask for its rows and uses the same padded lengths as the eager path, so it replays the kernels v1.2 ran
  eagerly: probabilities are bit-identical to the eager path, and the median latency per request is about half
  (67.9 ms to 31.9 ms on one A40). `/health` reports the number of recorded graphs (`cuda_graphs`).
- **Memory:** the server keeps only the option-letter rows of the output head (1.18 GiB less GPU memory allocated after
  loading; probabilities unchanged).
- **Serving temperature:** `decision_config.json` sets T = 0.9 for choice, yes/no and score questions (v1.2: T = 1.0).

An earlier internal v1.3 build (FULL_v5 weights) was not released; these weights supersede it.

## Results

v1.3 has not been measured on the JevBench leaderboard yet.

Our own reads, paired in the same harness (the bundled server or its offline equivalent and the JevBench `typesafe`
adapter), all at T = 0.9, against the earlier v1.3 build (FULL_v5 weights):

| Set | Earlier v1.3 build (Capability) | v1.3 (Capability) | v1.3 Intelligence / Calibration |
|---|---|---|---|
| 231 public JevBench items | 81.55 | 81.4 | 74.9 / 87.9 |
| Held-out proxy set (proxyA2) | 74.4 | 75.3 | |
| Independent JevBench replica (2,178 items) | 78.7 | **82.1** (+3.5, significant) | +3.4 / +3.5 over the earlier build (both significant) |

On the public items and our Proxy C set, v1.3 is level with the earlier build; the gain is on the independent
replica. On the public items, v1.2 at the same T = 0.9 scores Capability 80.3 (as shipped, at T = 1.0: 77.3), against
81.4 for v1.3. These are our measurements on public or internal items, not leaderboard scores.

### v1.2 (history)

On the JevBench leaderboard (release v1.6.34, as of 2026-10-20), Torchcast Decision 4B v1.2 was
**#1 among open-weight models by composite score**:

| Model | Composite | Intelligence | Calibration | Speed | Cost |
|---|---|---|---|---|---|
| **Torchcast Decision 4B v1.2** | **73.03** | 60.26 | 84.95 | 91.13 | 64.81 |
| H2O-Lightning-4B v1.1 | 72.52 | 60.04 | 89.95 | 92.61 | 60.30 |

Source: [benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models). v1.2 was measured on the
second regular v1.6 draw (anchor runs pending), so scores may be re-placed when the board's anchor
offsets are applied. These numbers are for v1.2 (tag `v1.2.0`), not v1.3.

## Run for evaluation

Linux with one NVIDIA GPU (tested on one H200 141 GB and one A40 48 GB, BF16). The weights repository contains the
serving package (`torchcast_decision/`) and its configuration; use both unchanged. As in v1.2, the full-attention layers
attend to the whole prompt in both directions (`decision_config.json`: `attention_mode` `noncausal_full_attention`);
only the bundled PyTorch server implements this. Do not serve the weights with vLLM, llama.cpp, MLX or a generic
`transformers` causal-LM pipeline: they apply a causal mask and give different probabilities.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.8.1" "flash-linear-attention==0.5.2" accelerate safetensors "huggingface_hub[cli]" \
    ninja packaging wheel setuptools
pip install --no-build-isolation "causal-conv1d>=1.7"

hf download torchcast-ai/torchcast-decision-4b --revision v1.3.0 --local-dir torchcast-decision-4b-v1.3
cd torchcast-decision-4b-v1.3
sha256sum -c SHA256SUMS
python -m torchcast_decision.check .                 # fast kernels must be active
python -m torchcast_decision.server --model . --host 127.0.0.1 --port 8011
```

The pinned artifact is tag `v1.3.0` of `torchcast-ai/torchcast-decision-4b`; weight SHA-256 values are in its
`SHA256SUMS`.

Check the served model name and that the CUDA graphs were recorded (`/health`: `"cuda_graphs"` above 0), then send a
warm-up request before timing:

```bash
curl -s http://127.0.0.1:8011/v1/models        # "name": "torchcast-decision-4b-v1.3"
curl -s http://127.0.0.1:8011/health           # "cuda_graphs": 52 on one H200
curl --fail http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer reports two charges for one order.","questions":{"decision":{"type":"choice","instructions":"Choose the support team.","criteria":{"billing":"Payments and refunds","technical":"Product malfunctions","other":"Other requests"}}}}'
```

For JevBench, use the `typesafe` adapter against `http://127.0.0.1:8011` with model name
`torchcast-decision-4b-v1.3`, for example from a JevBench checkout at commit
`7625ecb1b075cfe4d09d34141d353325825eea1d` (the runner requires the output paths to be outside the checkout):

```bash
J=datasets/public
python -m jevbench.cli run \
  --tasks $J/easy.jsonl,$J/original.jsonl,$J/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8011 --key-env "" \
  --model torchcast-decision-4b-v1.3 --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results ../runs/results.jsonl --raw-dir ../runs/raw --ledger ../runs/ledger.jsonl \
  --manifest ../runs/manifest.json --delay-s 0
```

The bundled `decision_config.json` applies a calibration temperature T = 0.9 to every question type and a yes/no
commit floor of 0.801 (neither changes the answer); prompts longer than 16,384 tokens keep the head and the tail of the state, as
`/health` reports. Requests are served one at a time. Choice questions take 2 or more options (more than 26 are
decided in groups); score questions take 2 to 10 levels (others are answered with HTTP 422). See the weights
repository's `NOTICE`.

The server listens on 127.0.0.1 by default; if the benchmark client runs on another machine, pass `--host 0.0.0.0`
and restrict access at the network level. This checkpoint has no published provider tariff; apply the benchmark's
documented estimated-cost method.

## Numerical reproducibility

Measured with this release on one NVIDIA H200 141 GB (CUDA 12.8, torch 2.9.1, transformers 5.8.1,
flash-linear-attention 0.5.2, Triton 3.5.1, BF16), all 231 JevBench public items, one question per request:

| Condition | Max probability difference | Predicted labels |
|---|---|---|
| CUDA graphs (v1.3 code) vs eager path (v1.2 code), same weights | 0 (bit-identical) | identical |
| Server restart, warm Triton cache | 0 (bit-identical) | identical |
| Fresh Triton cache (cold) vs warm | 0 (bit-identical) | identical |

The linear-attention layers run Triton kernels that are normally autotuned: on a fresh cache each kernel times its
candidate block configurations and keeps the fastest. Timing noise can select a different configuration, which changes
the BF16 summation order. The serving package therefore pins the selection (`torchcast_decision/_pin_triton.py`):
every candidate configuration is compiled and run once, so configurations that do not fit the GPU are skipped, and the
first valid configuration in the kernel's own list is used; no timings are read from or written to the on-disk cache.
Set `TC_TRITON_AUTOTUNE=1` to restore timing-based selection. Latency for these items on the H200, measured by the
benchmark client: p50 12.5 ms, p95 72.7 ms with the CUDA graphs (the same weights with the v1.2 code, which ran the
non-causal mode without CUDA graphs: p50 47.0 ms, p95 74.2 ms).

Different GPU models or software versions use different kernels and libraries (for example cuBLAS algorithms and
the valid Triton configurations) and can still give slightly different probabilities for the same request; compare
runs on the same GPU type and software.

## Licence

Model weights: Apache License 2.0. Base model: Qwen3.5-4B, Apache License 2.0, Copyright 2026 Alibaba Cloud. Serving package: Apache
License 2.0. See the licence files in the weights repository. This project is independent of TypeSafe AI and the
JevBench maintainers.
