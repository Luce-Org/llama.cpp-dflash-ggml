#include "expert-cache.h"

#include <algorithm>

namespace ggml_moe_stream {

expert_cache::expert_cache(int n_experts, size_t bytes_per_expert, int n_slots,
                           const void * host_experts, cudaStream_t copy_stream,
                           float decay, float hysteresis)
    : n_experts_(n_experts),
      bytes_(bytes_per_expert),
      n_slots_(std::min(n_slots, n_experts)),
      host_(static_cast<const char *>(host_experts)),
      copy_stream_(copy_stream),
      slot_of_(n_experts, -1),
      expert_in_(n_slots_, -1),
      protected_(n_experts, 0),
      ready_(n_slots_),
      planner_(n_experts, n_slots_, decay, hysteresis) {
    cudaMalloc(&dev_, (size_t) n_slots_ * bytes_);
    for (int s = 0; s < n_slots_; ++s) {
        cudaEventCreateWithFlags(&ready_[s], cudaEventDisableTiming);
    }
}

expert_cache::~expert_cache() {
    for (auto ev : ready_) cudaEventDestroy(ev);
    if (dev_) cudaFree(dev_);
}

std::vector<uint8_t> expert_cache::actual_resident_mask() const {
    std::vector<uint8_t> m(n_experts_, 0);
    for (int e = 0; e < n_experts_; ++e) if (slot_of_[e] >= 0) m[e] = 1;
    return m;
}

// Choose a slot to (re)use for `incoming`: empty first; else evict the
// lowest-usage (LFU) expert that is NOT in the current step's working set.
// Protecting the working set is what stops the thrash — evicting an expert that
// is needed this very step only to re-stream it (the +60-75% H2D copies we
// measured). Fall back to plain LFU only if every slot is working-set-pinned
// (working set larger than the cache, which shouldn't happen in practice).
int expert_cache::pick_victim_slot(int incoming) {
    for (int s = 0; s < n_slots_; ++s) if (expert_in_[s] == -1) return s;
    int best_slot = -1; double best_usage = 0.0;
    for (int s = 0; s < n_slots_; ++s) {
        const int e = expert_in_[s];
        if (e == incoming || protected_[e]) continue;
        const double u = planner_.usage(e);
        if (best_slot == -1 || u < best_usage) { best_usage = u; best_slot = s; }
    }
    if (best_slot != -1) return best_slot;
    for (int s = 0; s < n_slots_; ++s) {           // fallback: all slots pinned
        const int e = expert_in_[s];
        if (e == incoming) continue;
        const double u = planner_.usage(e);
        if (best_slot == -1 || u < best_usage) { best_usage = u; best_slot = s; }
    }
    return best_slot;
}

int expert_cache::bind(int e) {
    const int s   = pick_victim_slot(e);
    const int old = expert_in_[s];
    if (old >= 0) slot_of_[old] = -1;
    expert_in_[s] = e;
    slot_of_[e]   = s;
    return s;
}

const void * expert_cache::ensure_resident(int e, cudaStream_t compute_stream) {
    if (slot_of_[e] >= 0) {
        const int s = slot_of_[e];
        cudaStreamWaitEvent(compute_stream, ready_[s], 0); // honor in-flight prefetch
        return dev_ + (size_t) s * bytes_;
    }
    const int s = bind(e);
#ifndef STUB
    cudaMemcpyAsync(dev_ + (size_t) s * bytes_, host_ + (size_t) e * bytes_,
                    bytes_, cudaMemcpyHostToDevice, compute_stream);
    cudaEventRecord(ready_[s], compute_stream);
    ++h2d_copies;
#endif
    return dev_ + (size_t) s * bytes_;
}

void expert_cache::prefetch(const std::vector<int> & predicted, int max_prefetch) {
    const std::vector<int> plan = plan_prefetch(predicted, actual_resident_mask(), max_prefetch);
    for (int e : plan) {
        const int s = bind(e);
#ifndef STUB
        cudaMemcpyAsync(dev_ + (size_t) s * bytes_, host_ + (size_t) e * bytes_,
                        bytes_, cudaMemcpyHostToDevice, copy_stream_);
        cudaEventRecord(ready_[s], copy_stream_);
        ++h2d_copies;
#endif
    }
}

void expert_cache::observe(const int32_t * routed, int n) {
    // routed is this step's working set: protect it from eviction until next step.
    std::fill(protected_.begin(), protected_.end(), (uint8_t) 0);
    for (int i = 0; i < n; ++i) {
        const int e = routed[i];
        if (e >= 0 && e < n_experts_) protected_[e] = 1;
    }
    planner_.observe(routed, n);
}

const void * expert_cache::device_ptr(int e) const {
    return slot_of_[e] >= 0 ? dev_ + (size_t) slot_of_[e] * bytes_ : nullptr;
}

int expert_cache::resident_count() const {
    int c = 0; for (int s = 0; s < n_slots_; ++s) if (expert_in_[s] >= 0) ++c; return c;
}

} // namespace ggml_moe_stream
