// GPU red->green test for a FUSED MoE decode kernel.
//
// What it validates: for a small speculative-decode batch (T tokens, each routed
// to K experts) a SINGLE kernel launch does gather + on-the-fly 4-bit dequant +
// routed-weighted accumulation — replacing ggml's sort -> gather -> per-expert
// matmul. Correctness oracle: a CPU reference over the identical quantized data.
//
// Runs on the target silicon (RTX 3090, sm_86). Build:
//   nvcc -std=c++17 -arch=sm_86 -o t_moe.exe test-moe-fused-q4.cu
//
// Quant: real Q4_0 nibble packing (32 weights/block, half scale, w = d*(q-8)).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::printf("CUDA ERROR %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2);} } while(0)

struct block_q4_0 { __half d; uint8_t qs[16]; }; // 32 weights

// dims (small, exercises the path)
static const int T        = 5;    // spec-decode batch (tokens)
static const int K        = 4;    // experts routed per token (top-k)
static const int N_EXPERT = 16;
static const int N_ROWS   = 64;   // expert output dim
static const int N_COLS   = 128;  // contraction dim (mult of 32)
static const int BPR      = N_COLS / 32; // blocks per row

static inline float deq(const block_q4_0& b, int col) {
    const int bj = col >> 5, i = col & 31;
    const float d = __half2float(b.d);
    const int q = (i < 16) ? (b.qs[i] & 0x0F) : (b.qs[i - 16] >> 4);
    return d * (q - 8);
}

// ---------------- kernel under test ----------------
// grid: T*K blocks (one per (token, routed-expert)); blockDim.x threads stride rows.
__global__ void moe_fused_q4(const block_q4_0* __restrict__ W, const float* __restrict__ x,
                             const int* __restrict__ ids, const float* __restrict__ rw,
                             float* __restrict__ y, int n_rows, int n_cols, int k) {
#ifndef STUB
    const int pair = blockIdx.x;          // 0 .. T*K-1
    const int t    = pair / k;
    const int e    = ids[pair];
    const float w  = rw[pair];
    const int bpr  = n_cols >> 5;
    const block_q4_0* We = W + (size_t)e * n_rows * bpr;
    const float* xt = x + (size_t)t * n_cols;
    for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
        const block_q4_0* row = We + (size_t)r * bpr;
        float dot = 0.f;
        for (int bj = 0; bj < bpr; ++bj) {
            const block_q4_0 b = row[bj];
            const float d = __half2float(b.d);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int qlo = (b.qs[i] & 0x0F) - 8;
                const int qhi = (b.qs[i] >> 4)   - 8;
                dot += d * qlo * xt[bj*32 + i];
                dot += d * qhi * xt[bj*32 + i + 16];
            }
        }
        atomicAdd(&y[(size_t)t * n_rows + r], w * dot);
    }
#endif
}

int main() {
    std::srand(1234);
    std::vector<block_q4_0> W((size_t)N_EXPERT * N_ROWS * BPR);
    for (auto& b : W) { b.d = __float2half(0.01f + (std::rand()%90)*0.001f);
                        for (int i=0;i<16;++i) b.qs[i] = (uint8_t)(std::rand() & 0xFF); }
    std::vector<float> x(T * N_COLS); for (auto& v : x) v = (std::rand()%2001-1000)/1000.f;
    std::vector<int>   ids(T * K);    for (auto& v : ids) v = std::rand() % N_EXPERT;
    std::vector<float> rw(T * K);     for (auto& v : rw)  v = (std::rand()%1000)/1000.f;

    // CPU reference (double accumulation)
    std::vector<double> ref(T * N_ROWS, 0.0);
    for (int t=0;t<T;++t) for (int s=0;s<K;++s) {
        const int e = ids[t*K+s]; const double w = rw[t*K+s];
        const block_q4_0* We = W.data() + (size_t)e*N_ROWS*BPR;
        for (int r=0;r<N_ROWS;++r) { double dot=0;
            for (int c=0;c<N_COLS;++c) dot += (double)deq(We[(size_t)r*BPR + c/32], c) * x[t*N_COLS+c];
            ref[t*N_ROWS+r] += w*dot;
        }
    }

    block_q4_0* dW; float *dx,*drw,*dy; int* dids;
    CUDA_OK(cudaMalloc(&dW, W.size()*sizeof(block_q4_0)));
    CUDA_OK(cudaMalloc(&dx, x.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dids, ids.size()*sizeof(int)));
    CUDA_OK(cudaMalloc(&drw, rw.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, (size_t)T*N_ROWS*sizeof(float)));
    CUDA_OK(cudaMemcpy(dW, W.data(), W.size()*sizeof(block_q4_0), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dids, ids.data(), ids.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(drw, rw.data(), rw.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dy, 0, (size_t)T*N_ROWS*sizeof(float)));

    moe_fused_q4<<<T*K, 64>>>(dW, dx, dids, drw, dy, N_ROWS, N_COLS, K);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> y(T*N_ROWS);
    CUDA_OK(cudaMemcpy(y.data(), dy, y.size()*sizeof(float), cudaMemcpyDeviceToHost));

    double maxabs = 0, maxrel = 0;
    for (int i=0;i<T*N_ROWS;++i) { double a=std::fabs(y[i]-ref[i]); maxabs=std::max(maxabs,a);
        if (std::fabs(ref[i])>1e-6) maxrel=std::max(maxrel, a/std::fabs(ref[i])); }
    std::printf("max abs err=%.3e  max rel err=%.3e\n", maxabs, maxrel);
    const bool ok = maxabs < 1e-2;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
