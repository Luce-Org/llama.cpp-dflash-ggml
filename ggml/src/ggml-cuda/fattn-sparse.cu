// fattn-sparse.cu — ggml CUDA backend dispatch for GGML_OP_FLASH_ATTN_SPARSE.
//
// When a sparse kernel is registered via ggml_cuda_flash_attn_sparse_set_kernel,
// it is called with BF16 Q/K/V in flash_prefill_forward_bf16 layout:
//   Q[B, S, n_q_heads, D]   (contiguous, D fastest in C-row-major terms)
//   K[B, S, n_k_heads, D]
//   V[B, S, n_k_heads, D]
//   O[B, S, n_q_heads, D]
//
// ggml FA convention (column-major, ne[0] fastest):
//   Q src:  ne={D, S, H,  B}  =>  C-order [B][H][S][D]
//   K src:  ne={D, S, Hk, B}  =>  C-order [B][Hk][S][D]
//   V src:  ne={D, S, Hk, B}  =>  C-order [B][Hk][S][D]
//   O dst:  ne={D, S, H,  B}  =>  C-order [B][H][S][D]
//
// pFlash expects [B, S, H, D] row-major — S and H are swapped relative to ggml.
// A S<->H transpose is performed during type conversion (F32/F16 -> BF16) for
// Q/K/V inputs.  The output needs NO transpose: pFlash O[B,S,H,D] row-major
// already matches ggml dst ne={D,H,S,B} column-major, so a flat BF16->F32 copy
// suffices.
//
// When no kernel is registered, falls back to ggml_cuda_flash_attn_ext (dense FA).

#include "fattn-sparse.cuh"
#include "fattn.cuh"
#include "common.cuh"
#include "convert.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include <cstdlib>

static ggml_cuda_sparse_attn_fn_t s_sparse_kernel = nullptr;

void ggml_cuda_flash_attn_sparse_set_kernel(ggml_cuda_sparse_attn_fn_t fn) {
    s_sparse_kernel = fn;
}

// Convert F32 -> BF16 with (S,H) transpose.
// src: ggml [B,H,S,D] row-major.  dst: pFlash [B,S,H,D] row-major.
__global__ void k_f32_to_bf16_transpose_sh(
    const float * __restrict__ src, __nv_bfloat16 * __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    int src_idx = ((b * H + h) * S + s) * D + d;
    dst[idx] = __float2bfloat16(src[src_idx]);
}

// Convert F16 -> BF16 with (S,H) transpose.
// src: ggml [B,H,S,D] row-major.  dst: pFlash [B,S,H,D] row-major.
__global__ void k_f16_to_bf16_transpose_sh(
    const half * __restrict__ src, __nv_bfloat16 * __restrict__ dst,
    int B, int S, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % H;
    int s = (idx / (D * H)) % S;
    int b = idx / (D * H * S);
    int src_idx = ((b * H + h) * S + s) * D + d;
    dst[idx] = __float2bfloat16(__half2float(src[src_idx]));
}

// Flat BF16 -> F32 conversion (no transpose).
// pFlash output [B,S,H,D] row-major matches ggml dst ne={D,H,S,B} column-major —
// no transpose needed; a flat element-wise copy is correct.
__global__ void k_bf16_to_f32_flat(
    const __nv_bfloat16 * __restrict__ src, float * __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __bfloat162float(src[idx]);
}


static __device__ __forceinline__ float tree_q8_0_load_d128(const char * row, const int d) {
    const block_q8_0 * blocks = (const block_q8_0 *) row;
    const block_q8_0 & b = blocks[d / QK8_0];
    return __half2float(b.d) * (float)b.qs[d % QK8_0];
}

__global__ void k_flash_attn_tree_q8_0_d128(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        char       * __restrict__ dst,
        const float scale,
        const int32_t n_q,
        const int32_t n_head,
        const int32_t n_head_kv,
        const int32_t n_kv,
        const int32_t n_batch,
        const int64_t q_nb1,
        const int64_t q_nb2,
        const int64_t q_nb3,
        const int64_t k_nb1,
        const int64_t k_nb2,
        const int64_t k_nb3,
        const int64_t v_nb1,
        const int64_t v_nb2,
        const int64_t v_nb3,
        const int64_t m_nb1,
        const int64_t m_nb3,
        const int64_t d_nb1,
        const int64_t d_nb2,
        const int64_t d_nb3) {
    constexpr int D = 128;
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int d = threadIdx.x;
    if (q >= n_q || h >= n_head || b >= n_batch || d >= D) {
        return;
    }

    const int gqa = n_head / n_head_kv;
    const int hk = h / gqa;

    const char * q_row = Q + (int64_t)b*q_nb3 + (int64_t)h*q_nb2 + (int64_t)q*q_nb1;
    const char * k_base = K + (int64_t)b*k_nb3 + (int64_t)hk*k_nb2;
    const char * v_base = V + (int64_t)b*v_nb3 + (int64_t)hk*v_nb2;
    const char * m_row = mask + (int64_t)(b % n_batch)*m_nb3 + (int64_t)q*m_nb1;

    const float qd = ((const float *)q_row)[d];
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float ss = 0.0f;

    __shared__ float red[D];
    for (int k = 0; k < n_kv; ++k) {
        const half mh = ((const half *)m_row)[k];
        const float mask_v = __half2float(mh);
        if (mask_v <= -60000.0f) {
            continue;
        }

        const char * k_row = k_base + (int64_t)k*k_nb1;
        const float kd = tree_q8_0_load_d128(k_row, d);
        red[d] = qd * kd;
        __syncthreads();

        for (int stride = D/2; stride > 0; stride >>= 1) {
            if (d < stride) {
                red[d] += red[d + stride];
            }
            __syncthreads();
        }

        const float score = red[0] * scale + mask_v;
        const float m_new = fmaxf(m, score);
        const float old_scale = ss == 0.0f ? 0.0f : expf(m - m_new);
        const float p = expf(score - m_new);
        const char * v_row = v_base + (int64_t)k*v_nb1;
        const float vd = tree_q8_0_load_d128(v_row, d);
        acc = acc * old_scale + p * vd;
        ss = ss * old_scale + p;
        m = m_new;
        __syncthreads();
    }

    float * dst_row = (float *)(dst + (int64_t)b*d_nb3 + (int64_t)q*d_nb2 + (int64_t)h*d_nb1);
    dst_row[d] = ss > 0.0f ? acc / ss : 0.0f;
}

static bool ggml_cuda_flash_attn_tree_q8_0_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];
    if (!Q || !K || !V || !M || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || M->type != GGML_TYPE_F16) {
        return false;
    }
    if (Q->ne[0] != 128 || K->ne[0] != 128 || V->ne[0] != 128 || Q->ne[3] != 1) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || M->ne[0] < K->ne[1] || M->ne[1] < Q->ne[1]) {
        return false;
    }
    if (Q->nb[0] != (int64_t)sizeof(float) || dst->nb[0] != (int64_t)sizeof(float)) {
        return false;
    }
    return Q->ne[1] <= 65;
}

static void ggml_cuda_flash_attn_tree_q8_0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(ggml_cuda_flash_attn_tree_q8_0_supported(dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];

    float op_params[2];
    memcpy(op_params, dst->op_params, sizeof(op_params));
    const float scale = op_params[0];

    const dim3 grid((unsigned)Q->ne[1], (unsigned)Q->ne[2], (unsigned)Q->ne[3]);
    constexpr int block = 128;
    k_flash_attn_tree_q8_0_d128<<<grid, block, 0, ctx.stream()>>>(
        (const char *)Q->data,
        (const char *)K->data,
        (const char *)V->data,
        (const char *)M->data,
        (char *)dst->data,
        scale,
        (int32_t)Q->ne[1],
        (int32_t)Q->ne[2],
        (int32_t)K->ne[2],
        (int32_t)K->ne[1],
        (int32_t)Q->ne[3],
        Q->nb[1], Q->nb[2], Q->nb[3],
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        M->nb[1], M->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_flash_attn_tree_q8_0_d128_partials(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ partial_acc,
        float2     * __restrict__ partial_meta,
        const float scale,
        const int32_t n_q,
        const int32_t n_head,
        const int32_t n_head_kv,
        const int32_t n_kv,
        const int32_t n_batch,
        const int32_t n_chunks,
        const int32_t chunk_size,
        const int64_t q_nb1,
        const int64_t q_nb2,
        const int64_t q_nb3,
        const int64_t k_nb1,
        const int64_t k_nb2,
        const int64_t k_nb3,
        const int64_t v_nb1,
        const int64_t v_nb2,
        const int64_t v_nb3,
        const int64_t m_nb1,
        const int64_t m_nb3) {
    constexpr int D = 128;
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int z = blockIdx.z;
    const int c = z % n_chunks;
    const int b = z / n_chunks;
    const int d = threadIdx.x;
    if (q >= n_q || h >= n_head || b >= n_batch || d >= D) {
        return;
    }

    const int gqa = n_head / n_head_kv;
    const int hk = h / gqa;

    const char * q_row = Q + (int64_t)b*q_nb3 + (int64_t)h*q_nb2 + (int64_t)q*q_nb1;
    const char * k_base = K + (int64_t)b*k_nb3 + (int64_t)hk*k_nb2;
    const char * v_base = V + (int64_t)b*v_nb3 + (int64_t)hk*v_nb2;
    const char * m_row = mask + (int64_t)(b % n_batch)*m_nb3 + (int64_t)q*m_nb1;

    const float qd = ((const float *)q_row)[d];
    float acc = 0.0f;
    float m = -3.4028234663852886e38f;
    float ss = 0.0f;

    __shared__ float red[D];
    const int k_begin = c * chunk_size;
    const int k_end = (k_begin + chunk_size < n_kv) ? k_begin + chunk_size : n_kv;
    for (int k = k_begin; k < k_end; ++k) {
        const float mask_v = __half2float(((const half *)m_row)[k]);
        if (mask_v <= -60000.0f) {
            continue;
        }

        const char * k_row = k_base + (int64_t)k*k_nb1;
        const float kd = tree_q8_0_load_d128(k_row, d);
        red[d] = qd * kd;
        __syncthreads();

        for (int stride = D/2; stride > 0; stride >>= 1) {
            if (d < stride) {
                red[d] += red[d + stride];
            }
            __syncthreads();
        }

        const float score = red[0] * scale + mask_v;
        const float m_new = fmaxf(m, score);
        const float old_scale = ss == 0.0f ? 0.0f : expf(m - m_new);
        const float p = expf(score - m_new);
        const char * v_row = v_base + (int64_t)k*v_nb1;
        const float vd = tree_q8_0_load_d128(v_row, d);
        acc = acc * old_scale + p * vd;
        ss = ss * old_scale + p;
        m = m_new;
        __syncthreads();
    }

    const int64_t partial_row = ((((int64_t)b*n_head + h)*n_q + q)*n_chunks + c);
    partial_acc[partial_row*D + d] = acc;
    if (d == 0) {
        partial_meta[partial_row] = make_float2(m, ss);
    }
}

__global__ void k_flash_attn_tree_q8_0_d128_combine(
        const float  * __restrict__ partial_acc,
        const float2 * __restrict__ partial_meta,
        char         * __restrict__ dst,
        const int32_t n_q,
        const int32_t n_head,
        const int32_t n_batch,
        const int32_t n_chunks,
        const int64_t d_nb1,
        const int64_t d_nb2,
        const int64_t d_nb3) {
    constexpr int D = 128;
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int d = threadIdx.x;
    if (q >= n_q || h >= n_head || b >= n_batch || d >= D) {
        return;
    }

    float m = -3.4028234663852886e38f;
    for (int c = 0; c < n_chunks; ++c) {
        const int64_t row = ((((int64_t)b*n_head + h)*n_q + q)*n_chunks + c);
        const float2 ms = partial_meta[row];
        if (ms.y > 0.0f) {
            m = fmaxf(m, ms.x);
        }
    }

    float ss = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int64_t row = ((((int64_t)b*n_head + h)*n_q + q)*n_chunks + c);
        const float2 ms = partial_meta[row];
        if (ms.y <= 0.0f) {
            continue;
        }
        const float s = expf(ms.x - m);
        ss += ms.y * s;
        acc += partial_acc[row*D + d] * s;
    }

    float * dst_row = (float *)(dst + (int64_t)b*d_nb3 + (int64_t)q*d_nb2 + (int64_t)h*d_nb1);
    dst_row[d] = ss > 0.0f ? acc / ss : 0.0f;
}

static bool ggml_cuda_flash_attn_tree_q8_0_parallel_supported(const ggml_tensor * dst) {
    if (!ggml_cuda_flash_attn_tree_q8_0_supported(dst)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    return Q->ne[3] == 1 && K->ne[1] > 64;
}

static void ggml_cuda_flash_attn_tree_q8_0_parallel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(ggml_cuda_flash_attn_tree_q8_0_parallel_supported(dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * M = dst->src[3];

    float op_params[2];
    memcpy(op_params, dst->op_params, sizeof(op_params));
    const float scale = op_params[0];

    int chunk_size = 64;
    if (const char * env = std::getenv("DFLASH_LAGUNA_TREE_ATTN_Q8_PAR_CHUNK")) {
        const int v = std::atoi(env);
        if (v == 32 || v == 64 || v == 128 || v == 256) {
            chunk_size = v;
        }
    }
    if (const char * env = std::getenv("GGML_CUDA_TREE_ATTN_Q8_PAR_CHUNK")) {
        const int v = std::atoi(env);
        if (v == 32 || v == 64 || v == 128 || v == 256) {
            chunk_size = v;
        }
    }

    const int n_q = (int)Q->ne[1];
    const int n_head = (int)Q->ne[2];
    const int n_batch = (int)Q->ne[3];
    const int n_kv = (int)K->ne[1];
    const int n_chunks = (n_kv + chunk_size - 1) / chunk_size;
    const size_t n_rows = (size_t)n_batch * (size_t)n_head * (size_t)n_q * (size_t)n_chunks;

    float * partial_acc = nullptr;
    float2 * partial_meta = nullptr;
    CUDA_CHECK(cudaMallocAsync(&partial_acc, n_rows * 128 * sizeof(float), ctx.stream()));
    CUDA_CHECK(cudaMallocAsync(&partial_meta, n_rows * sizeof(float2), ctx.stream()));

    const dim3 grid_part((unsigned)n_q, (unsigned)n_head, (unsigned)(n_chunks*n_batch));
    constexpr int block = 128;
    k_flash_attn_tree_q8_0_d128_partials<<<grid_part, block, 0, ctx.stream()>>>(
        (const char *)Q->data,
        (const char *)K->data,
        (const char *)V->data,
        (const char *)M->data,
        partial_acc,
        partial_meta,
        scale,
        n_q,
        n_head,
        (int32_t)K->ne[2],
        n_kv,
        n_batch,
        n_chunks,
        chunk_size,
        Q->nb[1], Q->nb[2], Q->nb[3],
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        M->nb[1], M->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid_combine((unsigned)n_q, (unsigned)n_head, (unsigned)n_batch);
    k_flash_attn_tree_q8_0_d128_combine<<<grid_combine, block, 0, ctx.stream()>>>(
        partial_acc,
        partial_meta,
        (char *)dst->data,
        n_q,
        n_head,
        n_batch,
        n_chunks,
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(partial_acc, ctx.stream()));
    CUDA_CHECK(cudaFreeAsync(partial_meta, ctx.stream()));
}

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // K/V may be quantized (Q8_0, Q4_0, etc.).  If so, we dequantize them to F16
    // into temporary buffers before the S<->H transpose into BF16 pFlash layout.

    // Tree-attention carrier: src[3] is the exact dense mask fallback, while
    // src[4]/src[5] carry parent ids / positions for a future specialized
    // backend kernel. Until that kernel is installed, run dense masked FA so the
    // graph path is bit-equivalent to ggml_flash_attn_ext.
    const bool tree_mode = dst->src[4] != nullptr;
    const bool tree_vec_enabled = tree_mode &&
        (std::getenv("GGML_CUDA_TREE_ATTN_VEC") != nullptr ||
         std::getenv("DFLASH_LAGUNA_TREE_ATTN_VEC") != nullptr) &&
        std::getenv("GGML_CUDA_TREE_ATTN_VEC_DISABLE") == nullptr &&
        std::getenv("DFLASH_LAGUNA_TREE_ATTN_VEC_DISABLE") == nullptr &&
        ggml_cuda_flash_attn_ext_vec_supported(dst);
    const bool tree_vec_masked_enabled = tree_mode &&
        (std::getenv("GGML_CUDA_TREE_ATTN_VEC_MASKED") != nullptr ||
         std::getenv("DFLASH_LAGUNA_TREE_ATTN_VEC_MASKED") != nullptr) &&
        std::getenv("GGML_CUDA_TREE_ATTN_VEC_MASKED_DISABLE") == nullptr &&
        std::getenv("DFLASH_LAGUNA_TREE_ATTN_VEC_MASKED_DISABLE") == nullptr &&
        ggml_cuda_flash_attn_ext_vec_supported(dst);
    const bool tree_q8_enabled = tree_mode &&
        (std::getenv("GGML_CUDA_TREE_ATTN_Q8") != nullptr ||
         std::getenv("DFLASH_LAGUNA_TREE_ATTN_Q8") != nullptr) &&
        std::getenv("GGML_CUDA_TREE_ATTN_Q8_DISABLE") == nullptr &&
        std::getenv("DFLASH_LAGUNA_TREE_ATTN_Q8_DISABLE") == nullptr &&
        ggml_cuda_flash_attn_tree_q8_0_supported(dst);
    const bool tree_q8_parallel_enabled = tree_mode &&
        (std::getenv("GGML_CUDA_TREE_ATTN_Q8_PAR") != nullptr ||
         std::getenv("DFLASH_LAGUNA_TREE_ATTN_Q8_PAR") != nullptr) &&
        std::getenv("GGML_CUDA_TREE_ATTN_Q8_PAR_DISABLE") == nullptr &&
        std::getenv("DFLASH_LAGUNA_TREE_ATTN_Q8_PAR_DISABLE") == nullptr &&
        ggml_cuda_flash_attn_tree_q8_0_parallel_supported(dst);

    if (tree_q8_parallel_enabled) {
        ggml_cuda_flash_attn_tree_q8_0_parallel(ctx, dst);
        return;
    }
    if (tree_q8_enabled) {
        ggml_cuda_flash_attn_tree_q8_0(ctx, dst);
        return;
    }

    // When no sparse kernel is registered, fall back to dense FA.
    if (tree_mode || !s_sparse_kernel) {
        const enum ggml_op saved_op = dst->op;
        float saved_params[GGML_MAX_OP_PARAMS / sizeof(float)];
        memcpy(saved_params, dst->op_params, GGML_MAX_OP_PARAMS);
        ggml_tensor * saved_src3 = dst->src[3];
        ggml_tensor * saved_src4 = dst->src[4];
        ggml_tensor * saved_src5 = dst->src[5];

        float op_params[2];
        memcpy(op_params, dst->op_params, sizeof(op_params));
        const float scale = op_params[0];

        float ext_params[3] = { scale, 0.0f, 0.0f };
        dst->op = GGML_OP_FLASH_ATTN_EXT;
        memset(dst->op_params, 0, GGML_MAX_OP_PARAMS);
        memcpy(dst->op_params, ext_params, sizeof(ext_params));
        dst->src[3] = tree_mode ? saved_src3 : nullptr;
        dst->src[4] = nullptr;
        dst->src[5] = nullptr;

        if (tree_vec_masked_enabled) {
            ggml_cuda_flash_attn_ext_vec_masked_force(ctx, dst);
        } else if (tree_vec_enabled) {
            ggml_cuda_flash_attn_ext_vec_force(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext(ctx, dst);
        }

        dst->op = saved_op;
        memcpy(dst->op_params, saved_params, GGML_MAX_OP_PARAMS);
        dst->src[3] = saved_src3;
        dst->src[4] = saved_src4;
        dst->src[5] = saved_src5;
        return;
    }

    // Sparse path.
    // ggml src tensors (Q is F32, K/V are F16), ggml FA convention:
    //   Q: ne={D, S, H,  B}  C-order [B][H][S][D]
    //   K: ne={D, S, Hk, B}  C-order [B][Hk][S][D]
    //   V: ne={D, S, Hk, B}  C-order [B][Hk][S][D]
    // ggml dst:
    //   O: ne={D, S, H,  B}  C-order [B][H][S][D]  F32
    //
    // pFlash expects [B,S,H,D] — S and H are transposed relative to ggml for inputs.
    // Input (Q/K/V) type conversion kernels perform the S<->H transpose.
    // The output requires no transpose (flat BF16->F32 copy suffices).
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float op_params[2];
    memcpy(op_params, dst->op_params, sizeof(op_params));
    const float scale = op_params[0];
    const float alpha = op_params[1];

    const int D  = (int)Q->ne[0];
    const int S  = (int)Q->ne[1];  // seq_len (ggml FA convention: ne[1] = tokens)
    const int H  = (int)Q->ne[2];  // n_q_heads (ggml FA convention: ne[2] = heads)
    const int Hk = (int)K->ne[2];  // n_kv_heads
    const int B  = (int)(Q->ne[3] > 0 ? Q->ne[3] : 1);

    const int Q_n = B * H  * S * D;
    const int K_n = B * Hk * S * D;
    const int O_n = Q_n;

    cudaStream_t stream = ctx.stream();
    const int block = 256;

    // Allocate pFlash-layout BF16 buffers and convert with S<->H transpose for inputs.
    // Q: F32 ggml [B,H,S,D] -> BF16 pFlash [B,S,H,D]   (S<->H transpose)
    // K: F16 ggml [B,Hk,S,D] -> BF16 pFlash [B,S,Hk,D] (S<->H transpose; dequant if needed)
    // V: F16 ggml [B,Hk,S,D] -> BF16 pFlash [B,S,Hk,D] (S<->H transpose; dequant if needed)
    // O: pFlash [B,S,H,D] BF16 -> F32 into dst->data    (flat copy, no transpose)

    __nv_bfloat16 *Q_pf;
    __nv_bfloat16 *K_pf;
    __nv_bfloat16 *V_pf;
    __nv_bfloat16 *O_pf;

    CUDA_CHECK(cudaMallocAsync(&Q_pf, Q_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&K_pf, K_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&V_pf, K_n * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&O_pf, O_n * sizeof(__nv_bfloat16), stream));

    // Q: F32 ggml [B,H,S,D] -> BF16 pFlash [B,S,H,D]  (S<->H transpose)
    k_f32_to_bf16_transpose_sh<<<(Q_n + block - 1) / block, block, 0, stream>>>(
        (const float *)Q->data, Q_pf, B, S, H, D);

    // K: dequantize to F16 if needed, then transpose S<->H into BF16 pFlash layout.
    half * K_f16_buf = nullptr;
    const half * K_f16_src = nullptr;
    if (K->type == GGML_TYPE_F16) {
        K_f16_src = (const half *)K->data;
    } else {
        to_fp16_cuda_t to_fp16_k = ggml_get_to_fp16_cuda(K->type);
        GGML_ASSERT(to_fp16_k != nullptr && "no F16 dequant for K type");
        CUDA_CHECK(cudaMallocAsync(&K_f16_buf, (size_t)K_n * sizeof(half), stream));
        to_fp16_k(K->data, K_f16_buf, K_n, stream);
        K_f16_src = K_f16_buf;
    }
    k_f16_to_bf16_transpose_sh<<<(K_n + block - 1) / block, block, 0, stream>>>(
        K_f16_src, K_pf, B, S, Hk, D);

    // V: dequantize to F16 if needed, then transpose S<->H into BF16 pFlash layout.
    half * V_f16_buf = nullptr;
    const half * V_f16_src = nullptr;
    if (V->type == GGML_TYPE_F16) {
        V_f16_src = (const half *)V->data;
    } else {
        to_fp16_cuda_t to_fp16_v = ggml_get_to_fp16_cuda(V->type);
        GGML_ASSERT(to_fp16_v != nullptr && "no F16 dequant for V type");
        CUDA_CHECK(cudaMallocAsync(&V_f16_buf, (size_t)K_n * sizeof(half), stream));
        to_fp16_v(V->data, V_f16_buf, K_n, stream);
        V_f16_src = V_f16_buf;
    }
    k_f16_to_bf16_transpose_sh<<<(K_n + block - 1) / block, block, 0, stream>>>(
        V_f16_src, V_pf, B, S, Hk, D);

    // Call the registered pFlash kernel.
    // Expects Q[B,S,H,D], K[B,S,Hk,D], V[B,S,Hk,D], O[B,S,H,D] all BF16 contiguous.
    // The registered pFlash kernel launches on the default stream, but the
    // Q/K/V conversions above ran on ctx.stream().  Order them with events so
    // the default stream waits for the conversions and ctx.stream() waits for
    // pFlash's output, without stalling the host (ggml streams are non-blocking,
    // so the legacy default stream does not auto-synchronize with them).
    cudaEvent_t ev_conv;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_conv, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_conv, stream));
    CUDA_CHECK(cudaStreamWaitEvent(((cudaStream_t)0), ev_conv, 0));
    CUDA_CHECK(cudaEventDestroy(ev_conv));
    int err = s_sparse_kernel(Q_pf, K_pf, V_pf, O_pf,
                              B, S, H, Hk, D, scale, alpha);
    GGML_ASSERT(err == 0 && "sparse attention kernel failed");
    cudaEvent_t ev_pf;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_pf, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_pf, ((cudaStream_t)0)));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_pf, 0));
    CUDA_CHECK(cudaEventDestroy(ev_pf));

    // pFlash output [B,S,H,D] row-major matches ggml dst ne={D,H,S,B} column-major —
    // no transpose needed; flat BF16->F32 copy is correct.
    k_bf16_to_f32_flat<<<(O_n + block - 1) / block, block, 0, stream>>>(
        O_pf, (float *)dst->data, O_n);

    // Free temporary F16 dequant buffers (if allocated) then pFlash BF16 buffers.
    if (K_f16_buf) CUDA_CHECK(cudaFreeAsync(K_f16_buf, stream));
    if (V_f16_buf) CUDA_CHECK(cudaFreeAsync(V_f16_buf, stream));
    CUDA_CHECK(cudaFreeAsync(Q_pf, stream));
    CUDA_CHECK(cudaFreeAsync(K_pf, stream));
    CUDA_CHECK(cudaFreeAsync(V_pf, stream));
    CUDA_CHECK(cudaFreeAsync(O_pf, stream));
}
