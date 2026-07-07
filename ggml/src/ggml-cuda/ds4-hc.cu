#include "ds4-hc.cuh"

// Fused DeepSeek4 hyper-connection ops.
//
// mode 0 (pre):  src0 = mix [mix_dim]        (f32, from fn @ rms_norm(hc_state))
//                src1 = base [mix_dim]       (f32)
//                src2 = hc_state [n_embd*n_hc] (f32, raw residual streams)
//                dst  = [n_embd + mix_dim]:
//                       dst[0..n_embd)          = working vector (pre-mixed input)
//                       dst[n_embd..n_embd+mix) = split = {pre[n_hc], post[n_hc], comb[n_hc*n_hc]}
//                Math matches cpu_hc_sinkhorn + finish_hc_pre_from_mix_into in
//                deepseek4_graph.cpp (sigmoid gates + Sinkhorn-normalized combine).
//
// mode 1 (post): src0 = residual hc_state [n_embd*n_hc]
//                src1 = block_out [n_embd]
//                src2 = split [mix_dim] (view of a mode-0 dst tail)
//                dst  = new hc_state [n_embd*n_hc]:
//                       dst[h*n_embd+d] = post[h]*block_out[d]
//                                       + sum_src comb[h + src*n_hc] * residual[src*n_embd+d]
//
// mode 2 (out):  src0 = mix [n_hc]
//                src1 = base [n_hc]
//                src2 = hc_state [n_embd*n_hc]
//                dst  = [n_embd]: weights[h] = sigmoid(mix[h]*s0+base[h]) + 1e-6;
//                       dst[d] = sum_h weights[h]*hc_state[h*n_embd+d]

#define DS4_HC_SINKHORN_EPS 1.0e-6f
#define DS4_HC_MAX_HC 8
#define DS4_HC_MAX_MIX (2*DS4_HC_MAX_HC + DS4_HC_MAX_HC*DS4_HC_MAX_HC)

static __device__ __forceinline__ float ds4_hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ void ds4_hc_sinkhorn_split(
        const float * mix,
        const float * base,
        float         pre_scale,
        float         post_scale,
        float         comb_scale,
        int           n_hc,
        int           iters,
        float       * split) {
    for (int i = 0; i < n_hc; ++i) {
        split[i] = ds4_hc_sigmoid(mix[i] * pre_scale + base[i]) + DS4_HC_SINKHORN_EPS;
    }
    for (int i = 0; i < n_hc; ++i) {
        split[n_hc + i] = 2.0f * ds4_hc_sigmoid(mix[n_hc + i] * post_scale + base[n_hc + i]);
    }

    float c[DS4_HC_MAX_HC * DS4_HC_MAX_HC];
    for (int dst_i = 0; dst_i < n_hc; ++dst_i) {
        float row_max = -1.0e30f;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            const int idx = src_i + dst_i * n_hc;
            const float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
            c[idx] = v;
            row_max = v > row_max ? v : row_max;
        }
        float row_sum = 0.0f;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            const int idx = src_i + dst_i * n_hc;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            c[src_i + dst_i * n_hc] = c[src_i + dst_i * n_hc] * inv + DS4_HC_SINKHORN_EPS;
        }
    }
    for (int src_i = 0; src_i < n_hc; ++src_i) {
        float sum = 0.0f;
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) sum += c[src_i + dst_i * n_hc];
        const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) c[src_i + dst_i * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; ++iter) {
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) {
            float sum = 0.0f;
            for (int src_i = 0; src_i < n_hc; ++src_i) sum += c[src_i + dst_i * n_hc];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            for (int src_i = 0; src_i < n_hc; ++src_i) c[src_i + dst_i * n_hc] *= inv;
        }
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            float sum = 0.0f;
            for (int dst_i = 0; dst_i < n_hc; ++dst_i) sum += c[src_i + dst_i * n_hc];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            for (int dst_i = 0; dst_i < n_hc; ++dst_i) c[src_i + dst_i * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; ++i) {
        split[2 * n_hc + i] = c[i];
    }
}

static __global__ void ds4_hc_pre_kernel(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int   n_embd,
        int   n_hc,
        int   iters,
        float pre_scale,
        float post_scale,
        float comb_scale) {
    __shared__ float split[DS4_HC_MAX_MIX];
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    const int tid = threadIdx.x;

    if (tid == 0) {
        ds4_hc_sinkhorn_split(mix, base, pre_scale, post_scale, comb_scale, n_hc, iters, split);
        for (int i = 0; i < mix_dim; ++i) {
            dst[n_embd + i] = split[i];
        }
    }
    __syncthreads();

    for (int d = tid; d < n_embd; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; ++h) {
            acc += split[h] * hc_state[(size_t) h * n_embd + d];
        }
        dst[d] = acc;
    }
}

static __global__ void ds4_hc_post_kernel(
        const float * __restrict__ residual,
        const float * __restrict__ block_out,
        const float * __restrict__ split,
        float       * __restrict__ dst,
        int n_embd,
        int n_hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * n_hc;
    if (i >= total) {
        return;
    }
    const int h = i / n_embd;
    const int d = i - h * n_embd;
    const float * post = split + n_hc;
    const float * comb = split + 2 * n_hc;
    float acc = block_out[d] * post[h];
    for (int src = 0; src < n_hc; ++src) {
        acc += comb[h + src * n_hc] * residual[(size_t) src * n_embd + d];
    }
    dst[i] = acc;
}

static __global__ void ds4_hc_out_kernel(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int   n_embd,
        int   n_hc,
        float pre_scale) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }
    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        const float wgt = ds4_hc_sigmoid(mix[h] * pre_scale + base[h]) + DS4_HC_SINKHORN_EPS;
        acc += wgt * hc_state[(size_t) h * n_embd + d];
    }
    dst[d] = acc;
}

void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0 && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1 && src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2 && src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int mode   = ggml_get_op_params_i32(dst, 0);
    const int n_embd = ggml_get_op_params_i32(dst, 1);
    const int n_hc   = ggml_get_op_params_i32(dst, 2);

    GGML_ASSERT(n_hc > 0 && n_hc <= DS4_HC_MAX_HC);

    cudaStream_t stream = ctx.stream();

    switch (mode) {
        case 0: {
            const int   iters      = ggml_get_op_params_i32(dst, 3);
            const float pre_scale  = ggml_get_op_params_f32(dst, 4);
            const float post_scale = ggml_get_op_params_f32(dst, 5);
            const float comb_scale = ggml_get_op_params_f32(dst, 6);
            ds4_hc_pre_kernel<<<1, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (float *) dst->data,
                n_embd, n_hc, iters, pre_scale, post_scale, comb_scale);
        } break;
        case 1: {
            const int total = n_embd * n_hc;
            const int blocks = (total + 255) / 256;
            ds4_hc_post_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (float *) dst->data,
                n_embd, n_hc);
        } break;
        case 2: {
            const float pre_scale = ggml_get_op_params_f32(dst, 4);
            const int blocks = (n_embd + 255) / 256;
            ds4_hc_out_kernel<<<blocks, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (float *) dst->data,
                n_embd, n_hc, pre_scale);
        } break;
        default:
            GGML_ABORT("ds4_hc: unknown mode");
    }
}
