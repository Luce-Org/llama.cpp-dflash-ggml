#include "expert-stream-hooks.h"
#include "expert-cache.h"

#include <unordered_map>

namespace ggml_moe_stream {

static std::unordered_map<int, controller> & registry() {
    static std::unordered_map<int, controller> r;
    return r;
}

controller * moe_get(int layer) {
    auto & r = registry();
    auto it = r.find(layer);
    return it == r.end() ? nullptr : &it->second;
}

void moe_register(int layer, int n_experts, size_t bytes_per_expert, int n_slots,
                  const void * host_experts, cudaStream_t copy_stream,
                  float decay, float hysteresis) {
    auto & r = registry();
    controller & c = r[layer];
    delete c.cache;
    c.cache = new expert_cache(n_experts, bytes_per_expert, n_slots,
                               host_experts, copy_stream, decay, hysteresis);
}

void moe_reset() {
    auto & r = registry();
    for (auto & kv : r) delete kv.second.cache;
    r.clear();
}

void moe_observe(int layer, const int32_t * routed, int n) {
    if (controller * c = moe_get(layer)) c->cache->observe(routed, n);
}

void moe_prefetch(int layer, const std::vector<int> & draft_experts, int cap) {
    if (controller * c = moe_get(layer)) c->cache->prefetch(draft_experts, cap);
}

const void * moe_expert_ptr(int layer, int e, cudaStream_t stream) {
    controller * c = moe_get(layer);
    return c ? c->cache->ensure_resident(e, stream) : nullptr;
}

} // namespace ggml_moe_stream
