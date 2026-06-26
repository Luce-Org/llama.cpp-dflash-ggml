// GPU red->green test for the MoE integration controller/registry (task 1).
// Verifies hooks dispatch to the right per-layer cache and that unregistered
// layers are safe no-ops (so the mul_mat_id splice is opt-in / non-disruptive).
//   nvcc -std=c++17 -arch=sm_86 -allow-unsupported-compiler -o t_hooks.exe \
//        test-expert-hooks.cu ../ggml/src/ggml-cuda/expert-stream-hooks.cu \
//        ../ggml/src/ggml-cuda/expert-cache.cu ../ggml/src/ggml-cuda/expert-stream.cpp

#include "expert-stream-hooks.h"
#include "expert-cache.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace ggml_moe_stream;

static int g_total=0, g_fail=0;
#define CHECK(cond) do{ ++g_total; if(!(cond)){ ++g_fail; \
    std::printf("  FAIL [%d]: %s\n", __LINE__, #cond);} }while(0)

static const int NE=8; static const size_t BYTES=2048; static const int SLOTS=3;

int main() {
    uint8_t* host=nullptr; cudaMallocHost(&host,(size_t)NE*BYTES);
    for (int e=0;e<NE;++e) for (size_t i=0;i<BYTES;++i) host[e*BYTES+i]=(uint8_t)(e+1);
    cudaStream_t s, copy; cudaStreamCreate(&s); cudaStreamCreate(&copy);

    // Two MoE layers registered with independent caches.
    moe_register(/*layer*/0, NE, BYTES, SLOTS, host, copy);
    moe_register(/*layer*/1, NE, BYTES, SLOTS, host, copy);
    CHECK(moe_get(0) && moe_get(0)->cache);
    CHECK(moe_get(1) && moe_get(1)->cache);

    // expert_ptr on layer 0 makes it resident there only.
    const void* p = moe_expert_ptr(0, 3, s); cudaStreamSynchronize(s);
    CHECK(p != nullptr);
    CHECK(moe_get(0)->cache->is_resident(3));
    CHECK(!moe_get(1)->cache->is_resident(3));      // layer isolation

    // prefetch on layer 1 streams only layer 1.
    moe_prefetch(1, {5}, -1); cudaStreamSynchronize(copy);
    CHECK(moe_get(1)->cache->is_resident(5));
    CHECK(!moe_get(0)->cache->is_resident(5));

    // observe feeds only the addressed layer's policy (no crash, routed correctly).
    std::vector<int32_t> r = {2,2,2};
    for (int i=0;i<50;++i) moe_observe(0, r.data(), 3);

    // unregistered layer -> safe no-ops.
    CHECK(moe_get(99) == nullptr);
    CHECK(moe_expert_ptr(99, 0, s) == nullptr);
    moe_observe(99, r.data(), 3);                   // must not crash
    moe_prefetch(99, {1,2}, -1);                    // must not crash

    moe_reset();
    CHECK(moe_get(0) == nullptr);                   // cleaned up

    cudaStreamDestroy(s); cudaStreamDestroy(copy); cudaFreeHost(host);
    std::printf("\n%d/%d checks passed (%d failed)\n", g_total-g_fail, g_total, g_fail);
    return g_fail ? 1 : 0;
}
