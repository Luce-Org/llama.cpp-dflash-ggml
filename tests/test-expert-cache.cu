// GPU red->green test for the expert_cache mechanism (P1 prefetch + P3 residency).
// Synthetic experts: expert e's blob is all-bytes (e+1). We verify the RIGHT
// bytes land in a VRAM slot, capacity holds, eviction respects policy, and a
// speculative prefetch makes a later access a hit. Runs on RTX 3090 (sm_86).
//   nvcc -std=c++17 -arch=sm_86 -allow-unsupported-compiler -o t_cache.exe \
//        test-expert-cache.cu ../ggml/src/ggml-cuda/expert-cache.cu \
//        ../ggml/src/ggml-cuda/expert-stream.cpp

#include "expert-cache.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace ggml_moe_stream;

static int g_total = 0, g_fail = 0;
#define CHECK(cond) do { ++g_total; if(!(cond)){ ++g_fail; \
    std::printf("  FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond);} } while(0)

static const int    NE    = 8;
static const size_t BYTES = 4096;
static const int    SLOTS = 3;

// read back expert e's resident blob; true if all bytes == (e+1)
static bool slot_matches(expert_cache & c, int e) {
    const void * dp = c.device_ptr(e);
    if (!dp) return false;
    std::vector<uint8_t> h(BYTES);
    if (cudaMemcpy(h.data(), dp, BYTES, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (uint8_t b : h) if (b != (uint8_t)(e + 1)) return false;
    return true;
}

int main() {
    // host backing store: expert e -> all bytes (e+1), pinned for async copy
    uint8_t * host = nullptr;
    cudaMallocHost(&host, (size_t) NE * BYTES);
    for (int e = 0; e < NE; ++e)
        for (size_t i = 0; i < BYTES; ++i) host[(size_t) e * BYTES + i] = (uint8_t)(e + 1);

    cudaStream_t cs, copy; cudaStreamCreate(&cs); cudaStreamCreate(&copy);

    // T1: cold miss copies the correct expert into a slot
    {
        expert_cache c(NE, BYTES, SLOTS, host, copy);
        c.ensure_resident(3, cs);
        cudaStreamSynchronize(cs);
        CHECK(c.is_resident(3));
        CHECK(slot_matches(c, 3));
    }
    // T2: re-access is a hit on the same slot, data intact
    {
        expert_cache c(NE, BYTES, SLOTS, host, copy);
        const void * p1 = c.ensure_resident(2, cs); cudaStreamSynchronize(cs);
        const void * p2 = c.ensure_resident(2, cs); cudaStreamSynchronize(cs);
        CHECK(p1 == p2);
        CHECK(c.is_resident(2));
        CHECK(slot_matches(c, 2));
    }
    // T3: capacity never exceeded across many distinct experts
    {
        expert_cache c(NE, BYTES, SLOTS, host, copy);
        for (int e = 0; e < NE; ++e) { c.ensure_resident(e, cs); cudaStreamSynchronize(cs);
                                       CHECK(c.resident_count() <= SLOTS); }
        CHECK(c.resident_count() == SLOTS);
    }
    // T4: speculative prefetch on the copy stream makes a later access a hit
    {
        expert_cache c(NE, BYTES, SLOTS, host, copy);
        c.prefetch({6}, -1);
        cudaStreamSynchronize(copy);
        CHECK(c.is_resident(6));         // already streamed in
        CHECK(slot_matches(c, 6));       // and it's the right bytes
        const void * p = c.ensure_resident(6, cs); cudaStreamSynchronize(cs);
        CHECK(p == c.device_ptr(6));     // hit, no rebind
    }
    // T5: eviction prefers experts the policy doesn't want resident
    {
        expert_cache c(NE, BYTES, SLOTS, host, copy);
        std::vector<int32_t> hot = {0, 1, 2};
        for (int i = 0; i < 200; ++i) c.observe(hot.data(), 3);  // policy hot set {0,1,2}
        for (int e : {0,1,2}) { c.ensure_resident(e, cs); }       // fill slots with hot set
        c.ensure_resident(5, cs);                                 // one-off cold -> evicts someone
        for (int e : {0,1,2}) { c.ensure_resident(e, cs); }       // re-touch hot set
        cudaStreamSynchronize(cs);
        CHECK(c.is_resident(0) && c.is_resident(1) && c.is_resident(2));
        CHECK(!c.is_resident(5));         // the cold one-off got evicted, not the hot set
        CHECK(c.resident_count() == SLOTS);
        CHECK(slot_matches(c, 0) && slot_matches(c, 1) && slot_matches(c, 2));
    }

    cudaStreamDestroy(cs); cudaStreamDestroy(copy); cudaFreeHost(host);
    std::printf("\n%d/%d checks passed (%d failed)\n", g_total - g_fail, g_total, g_fail);
    return g_fail ? 1 : 0;
}
