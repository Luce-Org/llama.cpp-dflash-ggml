#pragma once

// CUDA mechanism that executes the expert-streaming policy (P1 prefetch + P3
// residency) for a memory-constrained MoE: experts live in host/unified memory,
// a fixed set of VRAM "slots" cache the hot ones, and cold experts are streamed
// in over the PCIe link on demand or — better — prefetched on a copy stream
// during the speculative draft pass so the verify pass finds them resident.
//
// This is the plumbing under expert-stream.h's pure-policy planners.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "expert-stream.h"

namespace ggml_moe_stream {

class expert_cache {
public:
    // n_experts        : total experts
    // bytes_per_expert : size of one expert's weight blob
    // n_slots          : VRAM-resident capacity (== residency budget)
    // host_experts     : base of all experts, contiguous, in (pinned) host memory
    // copy_stream      : stream used for speculative prefetch copies
    expert_cache(int n_experts, size_t bytes_per_expert, int n_slots,
                 const void * host_experts, cudaStream_t copy_stream,
                 float decay = 0.99f, float hysteresis = 1.10f);
    ~expert_cache();

    expert_cache(const expert_cache &) = delete;
    expert_cache & operator=(const expert_cache &) = delete;

    // Ensure expert e is resident; returns its device pointer. On a miss, streams
    // host->VRAM on compute_stream (evicting per policy). On a hit, makes
    // compute_stream wait on any in-flight prefetch for that slot.
    const void * ensure_resident(int e, cudaStream_t compute_stream);

    // P1: given the draft pass's predicted experts, async-copy the cold ones on
    // the copy stream (overlapped with draft compute). Non-blocking.
    void prefetch(const std::vector<int> & predicted, int max_prefetch);

    // P3: feed one decode step's routed experts to the residency policy.
    void observe(const int32_t * routed, int n);

    // introspection (tests / telemetry)
    bool         is_resident(int e) const { return slot_of_[e] >= 0; }
    const void * device_ptr(int e) const;
    int          resident_count() const;
    long         h2d_copies = 0;   // count of host->device expert streams issued (telemetry)

private:
    int                     pick_victim_slot(int incoming);
    std::vector<uint8_t>    actual_resident_mask() const;
    int                     bind(int e);                 // choose+claim a slot for e, return slot

    int                      n_experts_;
    size_t                   bytes_;
    int                      n_slots_;
    const char *             host_;
    cudaStream_t             copy_stream_;
    char *                   dev_ = nullptr;             // n_slots_ * bytes_
    std::vector<int>         slot_of_;                   // expert -> slot, or -1
    std::vector<int>         expert_in_;                 // slot   -> expert, or -1
    std::vector<uint8_t>     protected_;                 // expert -> 1 if in current step's working set
    std::vector<cudaEvent_t> ready_;                     // per-slot "copy done" event
    residency_planner        planner_;
};

} // namespace ggml_moe_stream
