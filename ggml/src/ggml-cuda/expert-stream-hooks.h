#pragma once

// Integration surface that wires the expert-streaming policy+mechanism into the
// ggml-cuda MoE path. One controller per (device, MoE layer): the model
// registers each layer's host-resident expert store once; then three hot-path
// hooks drive it:
//
//   moe_observe(layer, ids, n)            <- mul_mat_id, after routing is read
//                                            (ggml-cuda.cu, the ids_host loop)
//   moe_prefetch(layer, draft_ids, cap)   <- the DFlash draft pass, before verify
//   moe_expert_ptr(layer, e, stream)      <- the expert backing buffer, to resolve
//                                            a routed expert's device pointer
//
// All hooks are no-ops for unregistered layers, so the splice is safe/opt-in:
// with no registration the MoE path behaves exactly as before.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace ggml_moe_stream {

class expert_cache; // defined in expert-cache.h

struct controller { expert_cache * cache = nullptr; };

// registry
controller * moe_get(int layer);
void moe_register(int layer, int n_experts, size_t bytes_per_expert, int n_slots,
                  const void * host_experts, cudaStream_t copy_stream,
                  float decay = 0.99f, float hysteresis = 1.10f);
void moe_reset();

// hot-path hooks (safe no-ops if `layer` is not registered)
void         moe_observe(int layer, const int32_t * routed, int n);
void         moe_prefetch(int layer, const std::vector<int> & draft_experts, int cap);
const void * moe_expert_ptr(int layer, int e, cudaStream_t stream);

} // namespace ggml_moe_stream
