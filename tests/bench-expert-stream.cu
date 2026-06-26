// Real-hardware microbenchmark: speculative expert prefetch vs on-demand streaming,
// on the PCIe x4 link of a Lucebox. Models one Qwen3.6-35B-A3B MoE layer: experts
// live in pinned host memory, a resident subset is cached in VRAM, the rest stream.
//
// Decode-step model with speculative decoding:
//   draft phase  : a busy kernel ~ the draft-model forward (tunable duration)
//   verify phase : needs this step's top-k experts; cold ones must be in VRAM
//   BASELINE     : cold experts stream on-demand during verify (serialized after draft)
//   PREFETCH     : cold experts are prefetched on a copy stream DURING the draft phase,
//                  so verify finds them resident (the P1 win)
//
//   nvcc -std=c++17 -arch=sm_86 -I. -o bench bench-expert-stream.cu \
//        expert-cache.cu expert-stream.cpp

#include "expert-cache.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace ggml_moe_stream;

// ~35B-A3B single MoE layer
static const int    N_EXPERT   = 128;
static const int    TOPK       = 8;
static const int    RESIDENT   = 32;                  // VRAM-resident budget (experts)
static const size_t EXPERT_MB  = 10;                  // ~Q4_K bytes for one expert's FFN
static const size_t EXPERT_B   = EXPERT_MB << 20;
static const int    STEPS      = 200;
static const int    WARMUP     = 20;

// busy kernel ~ draft-model forward; iters tunes its duration
__global__ void draft_busy(volatile float* sink, long iters) {
    float a = 0.f; for (long i = 0; i < iters; ++i) a += __sinf(i * 1e-6f);
    if (threadIdx.x == 0) sink[blockIdx.x] = a;
}
// read the (just-streamed) expert bytes so the H2D actually matters; bandwidth-bound
// proxy for the expert matmul at small batch (coalesced + block reduction).
__global__ void touch_expert(const uint8_t* w, size_t n, float* out) {
    float s = 0;
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) s += w[i];
    __shared__ float sh[256];
    sh[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x/2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x+k]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

// Zipfian-ish routing with temporal locality: hot experts dominate, but the
// cold tail (what actually streams) shifts slowly — realistic for chat decode.
static std::vector<std::vector<int>> make_trace() {
    std::srand(20260626);
    std::vector<std::vector<int>> tr(STEPS);
    int bias = 0;
    for (int t = 0; t < STEPS; ++t) {
        if (t % 25 == 0) bias = std::rand() % N_EXPERT;   // slow topic drift
        std::vector<int> picks;
        while ((int)picks.size() < TOPK) {
            int e;
            if (std::rand()%100 < 70) e = std::rand()%RESIDENT;            // 70% hit hot set
            else e = (bias + std::rand()%N_EXPERT) % N_EXPERT;            // 30% cold tail
            if (std::find(picks.begin(),picks.end(),e)==picks.end()) picks.push_back(e);
        }
        tr[t] = picks;
    }
    return tr;
}

int main() {
    // pinned host expert store
    uint8_t* host = nullptr;
    if (cudaMallocHost(&host, (size_t)N_EXPERT*EXPERT_B) != cudaSuccess) { printf("pinned alloc failed\n"); return 2; }
    for (size_t i = 0; i < (size_t)N_EXPERT*EXPERT_B; i += 4096) host[i] = (uint8_t)i;

    // raw H2D bandwidth of this link
    { float* d; cudaMalloc(&d, EXPERT_B); cudaStream_t s; cudaStreamCreate(&s);
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
      cudaMemcpyAsync(d,host,EXPERT_B,cudaMemcpyHostToDevice,s); cudaStreamSynchronize(s); // warm
      cudaEventRecord(a,s); for(int i=0;i<20;i++) cudaMemcpyAsync(d,host,EXPERT_B,cudaMemcpyHostToDevice,s); cudaEventRecord(b,s);
      cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);
      printf("PCIe H2D bandwidth: %.2f GB/s  (expert=%zuMB -> %.2f ms each)\n",
             20.0*EXPERT_B/1e9/(ms/1e3), EXPERT_MB, ms/20.0);
      cudaFree(d); cudaStreamDestroy(s); }

    auto trace = make_trace();
    cudaStream_t cs, copy; cudaStreamCreate(&cs); cudaStreamCreate(&copy);
    float *sink, *acc; cudaMalloc(&sink, 1024*sizeof(float)); cudaMalloc(&acc, sizeof(float));

    // === VERIFY 1: does copy-on-copy-stream overlap compute-on-compute-stream over x4? ===
    {
        float* dbuf; cudaMalloc(&dbuf, 4*EXPERT_B);
        const long DI = 130000;                          // draft ~ the 4-expert copy time, so overlap is testable
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        // draft alone
        draft_busy<<<32,128,0,cs>>>(sink,DI); cudaDeviceSynchronize();
        cudaEventRecord(a,cs); draft_busy<<<32,128,0,cs>>>(sink,DI); cudaEventRecord(b,cs); cudaEventSynchronize(b);
        float t_draft; cudaEventElapsedTime(&t_draft,a,b);
        // copy alone (4 experts) on copy stream
        cudaEventRecord(a,copy); for(int i=0;i<4;i++) cudaMemcpyAsync(dbuf+i*EXPERT_B,host+(size_t)i*EXPERT_B,EXPERT_B,cudaMemcpyHostToDevice,copy); cudaEventRecord(b,copy); cudaEventSynchronize(b);
        float t_copy; cudaEventElapsedTime(&t_copy,a,b);
        // both concurrent (draft on cs, copy on copy) — time the wall
        cudaDeviceSynchronize(); cudaEvent_t w0,w1; cudaEventCreate(&w0); cudaEventCreate(&w1);
        cudaEventRecord(w0);
        draft_busy<<<32,128,0,cs>>>(sink,DI);
        for(int i=0;i<4;i++) cudaMemcpyAsync(dbuf+i*EXPERT_B,host+(size_t)i*EXPERT_B,EXPERT_B,cudaMemcpyHostToDevice,copy);
        cudaEventRecord(w1); cudaEventSynchronize(w1); cudaDeviceSynchronize();
        float t_both; cudaEventElapsedTime(&t_both,w0,w1);
        printf("OVERLAP test: draft=%.2fms  copy(4exp)=%.2fms  concurrent=%.2fms  -> %s (ideal=max=%.2f, serial=%.2f)\n",
               t_draft, t_copy, t_both, (t_both < 0.9f*(t_draft+t_copy) ? "OVERLAPS" : "NO OVERLAP"),
               fmaxf(t_draft,t_copy), t_draft+t_copy);
        cudaFree(dbuf);
    }

    // --- measure the real cache-miss (cold-expert) rate after warmup ---
    {
        expert_cache cache(N_EXPERT, EXPERT_B, RESIDENT, host, copy);
        long misses = 0, used = 0;
        for (int t = 0; t < STEPS; ++t) {
            cache.observe(trace[t].data(), (int)trace[t].size());   // score routing first (LFU protects working set)
            for (int e : trace[t]) { if (t >= WARMUP) { used++; if (!cache.is_resident(e)) misses++; }
                                     cache.ensure_resident(e, cs); }
        }
        cudaStreamSynchronize(cs);
        double cold_per_step = (double)misses / (STEPS - WARMUP);
        double cold_h2d_ms   = cold_per_step * (EXPERT_B / (6.52e9)) * 1e3; // at measured BW
        printf("steady-state: %.2f cold experts/step (of top-%d) -> %.2f ms/layer-step of x4 streaming\n",
               cold_per_step, TOPK, cold_h2d_ms);
    }

    // --- calibrate draft_busy: iters -> ms ---
    auto draft_ms = [&](long iters)->double {
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        draft_busy<<<32,128,0,cs>>>(sink, iters); cudaStreamSynchronize(cs); // warm
        cudaEventRecord(a,cs); draft_busy<<<32,128,0,cs>>>(sink, iters); cudaEventRecord(b,cs);
        cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return ms; };
    long base_iters = 1000000; double base_ms = draft_ms(base_iters);
    auto iters_for = [&](double ms){ return (long)(base_iters * ms / base_ms); };

    auto run = [&](bool prefetch, long draft_iters, long& copies)->double {
        expert_cache cache(N_EXPERT, EXPERT_B, RESIDENT, host, copy);
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        long c0 = 0;
        for (int t = 0; t < STEPS; ++t) {
            if (t == WARMUP) { cudaDeviceSynchronize(); cudaEventRecord(t0); c0 = cache.h2d_copies; }
            cache.observe(trace[t].data(), (int)trace[t].size());      // score routing first (LFU protects working set)
            if (prefetch) cache.prefetch(trace[t], RESIDENT);          // overlap with draft below
            draft_busy<<<32,128,0,cs>>>(sink, draft_iters);            // draft-model forward
            for (int e : trace[t]) {
                const void* w = cache.ensure_resident(e, cs);
                touch_expert<<<512,256,0,cs>>>((const uint8_t*)w, EXPERT_B, acc);
            }
            cudaStreamSynchronize(cs);
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        copies = cache.h2d_copies - c0;                                // H2D issued in the timed window
        float ms; cudaEventElapsedTime(&ms, t0, t1); return ms / (STEPS - WARMUP);
    };

    printf("\n--- one 35B-A3B MoE layer (%d experts, top-%d, %d resident, %zuMB/expert) ---\n",
           N_EXPERT, TOPK, RESIDENT, EXPERT_MB);
    printf("%-13s %11s %11s %9s %7s %14s\n", "draft window", "baseline ms", "prefetch ms", "saved %", "", "H2D copies b/p");
    for (double d : {1.0, 2.0, 4.0, 8.0}) {        // draft-forward duration per layer-step (ms)
        long it = iters_for(d), bc = 0, pc = 0;
        double base = run(false, it, bc), pref = run(true, it, pc);
        printf("%6.1f ms     %11.3f %11.3f %8.1f%%         %7ld / %ld\n", d, base, pref, 100.0*(base-pref)/base, bc, pc);
    }
    printf("(if prefetch H2D copies >> baseline, the prefetch policy is THRASHING the cache)\n");
    cudaFreeHost(host);
    return 0;
}
