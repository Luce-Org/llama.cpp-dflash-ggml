# MoE expert streaming — integration guide

Components (all unit-tested on RTX 3090, `sm_86`):

| File | Role | Test |
|------|------|------|
| `expert-stream.{h,cpp}` | P3 residency + P1 prefetch *policy* (pure C++) | `tests/test-expert-stream.cpp` — 15/15 |
| `expert-cache.{h,cu}` | VRAM slot cache: streams, events, eviction *mechanism* | `tests/test-expert-cache.cu` — 21/21 |
| `expert-stream-hooks.{h,cu}` | per-layer registry + hot-path hooks *integration* | `tests/test-expert-hooks.cu` — 10/10 |
| `moe-fused` kernels | fused gather+dequant+route, Q4_0 & Q4_K | `tests/test-moe-fused-q4{,k}.cu` — PASS |

The hooks are **no-ops until a layer is registered**, so every splice below is
opt-in: with no registration the MoE path is byte-for-byte unchanged.

## Splice points

### 0. Registration — model load (once per MoE layer)
Experts for the layer must live in **pinned host memory** (the streaming source).
```cpp
ggml_moe_stream::moe_register(layer_id, n_experts, bytes_per_expert,
                              n_resident_slots /* = VRAM budget */,
                              host_expert_base /* pinned */, copy_stream);
```
`n_resident_slots` is the "13.3 GiB" budget expressed in experts.

### 1. observe() — `ggml-cuda.cu`, `ggml_cuda_mul_mat_id`
The routing is already read to host at the `ids_host` loop (~L2613–2629). Right
after that loop, feed the routed experts to the residency policy:
```cpp
// after ids_host is populated and tokens_per_expert computed
std::vector<int32_t> routed;
for (int64_t i = 0; i < ne02; ++i)
    if (tokens_per_expert[i] > 0) routed.push_back((int32_t) i);
ggml_moe_stream::moe_observe(layer_id, routed.data(), (int) routed.size());
```
The fast paths (mmvq/mmq/mmf at ~L2562–2582) carry `ids` too; observe can be
hoisted to a shared helper covering both. `layer_id` comes from tagging the
expert tensor at load (e.g. `tensor->op_params` or a `->extra` field).

### 2. ensure_resident() — the expert backing buffer (the streaming win)
Where the per-expert weight pointer is computed (`src0->data + i02*nb[2]`),
resolve cold experts through the cache instead:
```cpp
const void * w = ggml_moe_stream::moe_expert_ptr(layer_id, (int) i02, stream);
// use `w` as expert i02's weight base; the cache has streamed it H2D and the
// returned stream waits on the in-flight copy. NULL => layer not registered,
// fall back to src0->data + i02*nb[2] (unchanged behavior).
```
This is the only edit that changes data movement; it belongs at the buffer/op
seam, not inside the matmul kernels.

### 3. prefetch() — the DFlash draft pass (hides the x4 latency)
After the draft model predicts the next tokens' routing, before the verify pass:
```cpp
// predicted = expert ids the draft expects the verify pass to use, this layer
ggml_moe_stream::moe_prefetch(layer_id, predicted, /*cap*/ slots_free);
```
Issued on `copy_stream`, overlapped with draft compute; by verify time the
experts are resident (proven by `test-expert-cache.cu` T4).

## Validation gate (deferred — GPU busy)
1. Build ggml-cuda with these TUs added to `ggml/src/ggml-cuda/CMakeLists.txt`.
2. Register layers at load; point expert tensors at the pinned host store.
3. `nsys` a 35B-A3B decode before/after: expect the PCIe-x4 H2D timeline to
   de-saturate and tok/s to rise. That number is the real proof.
