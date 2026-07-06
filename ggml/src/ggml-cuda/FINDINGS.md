# MoE expert-streaming: what's proven, what's not (real-hardware findings)

Measured on a real **Lucebox (lucebox3)** — RTX 3090 (`sm_86`), Ryzen AI MAX+ 395,
PCIe Gen4 **x4** (~6.5 GB/s measured), native CUDA 12.0 build of `dflash_server`,
Qwen3.6-35B-A3B (qwen35moe, 40 layers / 256 experts), Spark expert offload.

This PR's components stand **independently of the speculative-prefetch idea**.
Below is the honest scope: what the hardware proved, and where the prefetch
thesis remains open.

## 1. The two regimes (decisive)

Same 800-token decode, Spark residency varied:

| budget | resident experts | PCIe-RX (H2D) | x4 util | GPU sm | decode |
|---|---|---|---|---|---|
| 22 GiB (comfortable) | 9389 / 10240 (91%) | ~120 MB/s | ~2% | ~48% | 23.9 s (~33.5 tok/s) |
| **8 GiB (tight)** | ~half cold | **~3,000 MB/s, peaks 5,180** | **~50–80%** | 73% | **44.7 s (~18 tok/s)** |

**Expert streaming is the consumer-VRAM bottleneck *only when the model doesn't
fit*.** At comfortable residency the x4 link is idle (~2%) and decode is
autoregressive/compute-bound (GPU ~50% idle at batch=1). Force the budget tight
and the x4 link **saturates and decode slows 1.9×** — the regime a 24 GB card
hits with a model too big to mostly-fit (86 GB DeepSeek-V4-Flash, 78 GB
Qwen3.5-122B) or long context. That tight regime is the only place expert
prefetch has a target.

## 2. TQ3_0 vs Q4_0 KV cache

Same 22 GiB residency, identical decode:

| KV cache | decode | ≈ tok/s |
|---|---|---|
| `tq3_0` (default auto) | 26.9 s | 30 |
| `q4_0` | **23.9 s** | **33.5** |

**TQ3_0 KV costs ~11% here** — a real, free win to switch to Q4 KV at this
config (GPU stays ~50% idle either way, so KV is a tax, not the wall).

## 3. The prefetch thesis: mechanism-proven, recall-untested

- **Mechanism (this PR):** `expert_cache` + `plan_prefetch` validated in
  `tests/bench-expert-stream.cu` on the real x4 link. With the **working-set-
  protected eviction** (see §4) prefetch streams *exactly* the same bytes as
  on-demand (443 == 443 H2D copies) and overlaps them with the draft window:
  **+25–38% per layer-step** when the model doesn't fit. The mechanism works.
- **Recall (open):** whether a *draft model* predicts the next token's experts
  at high enough recall to drive that prefetch could **not** be measured here.
  The available DFlash drafter (`modal-labs/Qwen3.6-35B-A3B-DFlash`, converted to
  GGUF) loads and `ddtree=ON`, but delivers only ~7% speedup (36 vs 33.5 tok/s)
  — it ships without the Domino aux heads, and the DDTree's variable accept-
  length churns the CUDA graph (constant warmup/reset). Speculation never
  reaches its 2–3×, so draft→expert recall stays unmeasured.

Net: prefetch's *target* is real (§1) and its *mechanism* is proven; the
*predictor* (draft) is the remaining unknown, gated on a fully-working DFlash
draft on this build.

## 4. What lands here regardless of the draft

1. **Fused Q4_0/Q4_K MoE kernel** (`tests/test-moe-fused-q4{,k}.cu`):
   gather + on-the-fly dequant + routed accumulation in one launch, validated to
   `rel 6.4e-6` vs a CPU oracle on `sm_86`. Q4_K dequant mirrors ggml
   `get_scale_min_k4` bit-for-bit.
2. **Working-set-protected LFU eviction** (`expert-cache.cu`): never evict an
   expert in the current step's working set. Found by *measuring* H2D copy
   counts (naive prefetch issued +64% copies = thrash); the fix drops it to
   parity. This is a transferable improvement for **any** bounded expert cache,
   including Spark's.
3. **This characterization** — the regime boundary (§1) and the TQ3 tax (§2)
   are reusable facts for tuning expert offload on consumer hardware.

Policy/residency (`residency_planner`) overlaps Spark's shipped calibrated
placement + bounded cache; it's a clean, unit-tested reference, not a
replacement. Cross-platform: suite is 5/5 green on both Windows (MSVC) and the
Lucebox (native gcc/nvcc), `sm_86`.
