#pragma once

// Pure-C++ policy layer for MoE expert streaming on memory-constrained GPUs.
//
// Context: on a Lucebox (RTX 3090, 24 GB) the experts of a large MoE model
// (e.g. Qwen3.6 35B-A3B) live in host/unified memory and are streamed into VRAM
// over a PCIe Gen4 x4 link (~8 GB/s) on demand. That link, not compute, is the
// decode bottleneck. Two policies cut the traffic:
//
//   * residency_planner (P3): pick which experts stay pinned in VRAM, by decayed
//     usage frequency, with hysteresis so the resident set does not thrash.
//   * plan_prefetch     (P1): given the speculative draft pass's predicted expert
//     routing, list the cold experts to prefetch (overlapped with draft compute)
//     so the verify pass finds them already resident.
//
// No CUDA dependency on purpose: this is the decision logic, unit-testable on
// CPU. The CUDA mechanism (ggml_cuda_pool_vmm + a copy stream) executes the plans.

#include <cstdint>
#include <vector>

namespace ggml_moe_stream {

// P3 — steady-state residency policy for one MoE layer's expert pool.
class residency_planner {
public:
    // n_experts : total experts in the pool
    // budget    : how many may stay VRAM-resident (budget < n_experts)
    // decay     : per-step multiplicative decay applied to usage scores (1.0 = no decay)
    // hysteresis: a non-resident challenger must beat the weakest resident's score
    //             by this factor before it displaces it (>= 1.0; 1.0 = no hysteresis)
    residency_planner(int n_experts, int budget, float decay = 0.99f, float hysteresis = 1.10f);

    // Account for one decode step: decay all scores, add 1 to each used expert,
    // then recompute the resident set. ids may contain repeats (per-token routing).
    void observe(const int32_t * expert_ids, int n);

    const std::vector<uint8_t> & resident_mask() const { return resident_; } // size n_experts, 1=resident
    std::vector<int>             resident_set()  const;                       // sorted resident ids
    bool                         is_resident(int e) const;
    double                       usage(int e) const;                          // decayed use score (for LFU eviction)

private:
    void update_residency();

    int                 n_experts_;
    int                 budget_;
    double              decay_;
    double              hysteresis_;
    std::vector<double> score_;     // size n_experts
    std::vector<uint8_t> resident_; // size n_experts
};

// P1 — speculative prefetch plan.
// predicted_experts : expert ids the draft pass predicts the verify pass will use
//                     (may contain repeats; more repeats = higher demand)
// resident_mask     : current residency (from residency_planner); resident experts
//                     are skipped (already in VRAM)
// max_prefetch      : cap on experts to prefetch this window (models the x4 budget
//                     during the draft compute window); < 0 means unlimited
// returns cold expert ids ordered by demand (desc), ties by id (asc).
std::vector<int> plan_prefetch(const std::vector<int>  & predicted_experts,
                               const std::vector<uint8_t> & resident_mask,
                               int max_prefetch);

} // namespace ggml_moe_stream
