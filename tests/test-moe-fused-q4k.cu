// GPU red->green test: fused MoE decode kernel for Q4_K (production quant).
// Dequant mirrors ggml's dequantize_row_q4_K + get_scale_min_k4 bit-for-bit, so
// the same formula runs in the CPU oracle and on-the-fly in the kernel.
//   nvcc -std=c++17 -arch=sm_86 -allow-unsupported-compiler -o t_moek.exe test-moe-fused-q4k.cu

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    std::printf("CUDA ERROR %s @ %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2);} } while(0)

#define QK_K 256
#define K_SCALE_SIZE 12
struct block_q4_K { __half d; __half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; };
static_assert(sizeof(block_q4_K) == 2*sizeof(__half) + K_SCALE_SIZE + QK_K/2, "q4_K size");

__host__ __device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >>  4) | ((q[j]   >> 6) << 4); }
}

// dot of one Q4_K-quantized weight row (n_cols values) with x; mirrors ggml dequant.
template<bool DEVICE>
__host__ __device__ inline float q4k_row_dot(const block_q4_K* row, const float* x, int n_cols) {
    float dot = 0.f;
    const int nb = n_cols / QK_K;
    for (int sb = 0; sb < nb; ++sb) {
        const block_q4_K b = row[sb];
        const float d    = __half2float(b.d);
        const float dmin = __half2float(b.dmin);
        const float* xb  = x + sb * QK_K;
        const uint8_t* q = b.qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, b.scales, &sc, &m); const float d1 = d*sc, m1 = dmin*m;
            get_scale_min_k4(is + 1, b.scales, &sc, &m); const float d2 = d*sc, m2 = dmin*m;
            for (int l = 0; l < 32; ++l) dot += (d1 * (q[l] & 0xF) - m1) * xb[j + l];
            for (int l = 0; l < 32; ++l) dot += (d2 * (q[l] >>  4) - m2) * xb[j + 32 + l];
            q += 32; is += 2;
        }
    }
    return dot;
}

static const int T=4, K=3, N_EXPERT=8, N_ROWS=32, N_COLS=512; // 2 super-blocks/row
static const int BPR = N_COLS / QK_K;

__global__ void moe_fused_q4k(const block_q4_K* __restrict__ W, const float* __restrict__ x,
                              const int* __restrict__ ids, const float* __restrict__ rw,
                              float* __restrict__ y, int n_rows, int n_cols, int k) {
#ifndef STUB
    const int pair = blockIdx.x; const int t = pair / k;
    const int e = ids[pair]; const float w = rw[pair];
    const int bpr = n_cols / QK_K;
    const block_q4_K* We = W + (size_t)e * n_rows * bpr;
    const float* xt = x + (size_t)t * n_cols;
    for (int r = threadIdx.x; r < n_rows; r += blockDim.x) {
        const float dot = q4k_row_dot<true>(We + (size_t)r * bpr, xt, n_cols);
        atomicAdd(&y[(size_t)t * n_rows + r], w * dot);
    }
#endif
}

int main() {
    std::srand(7);
    std::vector<block_q4_K> W((size_t)N_EXPERT*N_ROWS*BPR);
    for (auto& b : W) {
        b.d = __float2half(0.02f + (std::rand()%80)*0.001f);
        b.dmin = __float2half(0.005f + (std::rand()%50)*0.0002f);
        for (int i=0;i<K_SCALE_SIZE;++i) b.scales[i] = (uint8_t)(std::rand()&0xFF);
        for (int i=0;i<QK_K/2;++i)       b.qs[i]     = (uint8_t)(std::rand()&0xFF);
    }
    std::vector<float> x(T*N_COLS); for (auto& v:x) v=(std::rand()%2001-1000)/1000.f;
    std::vector<int>   ids(T*K);    for (auto& v:ids) v=std::rand()%N_EXPERT;
    std::vector<float> rw(T*K);     for (auto& v:rw)  v=(std::rand()%1000)/1000.f;

    std::vector<double> ref(T*N_ROWS,0.0);
    for (int t=0;t<T;++t) for (int s=0;s<K;++s) {
        const int e=ids[t*K+s]; const double w=rw[t*K+s];
        const block_q4_K* We=W.data()+(size_t)e*N_ROWS*BPR;
        for (int r=0;r<N_ROWS;++r)
            ref[t*N_ROWS+r] += w * (double) q4k_row_dot<false>(We+(size_t)r*BPR, x.data()+t*N_COLS, N_COLS);
    }

    block_q4_K* dW; float *dx,*drw,*dy; int* dids;
    CUDA_OK(cudaMalloc(&dW, W.size()*sizeof(block_q4_K)));
    CUDA_OK(cudaMalloc(&dx, x.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dids, ids.size()*sizeof(int)));
    CUDA_OK(cudaMalloc(&drw, rw.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dy, (size_t)T*N_ROWS*sizeof(float)));
    CUDA_OK(cudaMemcpy(dW,W.data(),W.size()*sizeof(block_q4_K),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dids,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(drw,rw.data(),rw.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dy,0,(size_t)T*N_ROWS*sizeof(float)));

    moe_fused_q4k<<<T*K, 32>>>(dW,dx,dids,drw,dy,N_ROWS,N_COLS,K);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> y(T*N_ROWS);
    CUDA_OK(cudaMemcpy(y.data(),dy,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double maxabs=0,maxrel=0;
    for (int i=0;i<T*N_ROWS;++i){ double a=std::fabs(y[i]-ref[i]); maxabs=std::max(maxabs,a);
        if (std::fabs(ref[i])>1e-6) maxrel=std::max(maxrel,a/std::fabs(ref[i])); }
    std::printf("max abs err=%.3e  max rel err=%.3e\n", maxabs, maxrel);
    const bool ok = maxrel < 2e-3;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
