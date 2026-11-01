// ATTENTION V2: RDNA2-tuned attention (fp32 in/out, fp16 shared-memory tiles).
//
// Replaces the generic manual-attention path (separate mul_mat + softmax + mul_mat,
// all F32) for the common multi-head self-attention case.
//
// Layout (as produced by stable-diffusion.cpp):
//   q : [D, L_q, H*N]   contiguous F32
//   k : [D, L_k, H*N]   contiguous F32
//   v : [D, L_k, H*N]   contiguous F32
//   o : [D, H, L_q, N]  contiguous F32
//
// Measured on gfx1030: 8.5-8.8 TFLOPS for H=56, D=128, N=2430/5670
// (see kernels/attn_v2.cpp for the standalone benchmark).

#pragma once

#include "common.cuh"

#define ATTN_V2_BM 128
#define ATTN_V2_BN 32
#define ATTN_V2_R  8
#define ATTN_V2_THREADS 256

struct half4v { __half2 a, b; };

// v_dot2c_f32_f16: two fp16 products accumulated into one fp32 register.
static __device__ __forceinline__ float attn_v2_dot2(__half2 a, __half2 b, float c) {
#if defined(__AMDGCN__) && (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
                           defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx90a__))
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    const float2 af = __half22float2(a);
    const float2 bf = __half22float2(b);
    return c + af.x * bf.x + af.y * bf.y;
#endif
}

template <int BM, int BN, int R, int NTHREADS>
static __global__ void attn_v2_kernel(const float * __restrict__ Q,
                                      const float * __restrict__ K,
                                      const float * __restrict__ V,
                                      float       * __restrict__ O,
                                      int L_q, int L_k, int H, int N, int D, float scale) {
    constexpr int ROW_GROUPS = BM / R;
    constexpr int TPG        = NTHREADS / ROW_GROUPS;
    constexpr int KEYS       = BN / TPG;
    constexpr int COLS       = 128 / TPG;   // D is 128 here

    __shared__ __half sQ[BM][128];
    __shared__ __half sK[BN][128];
    __shared__ __half sV[BN][128];
    __shared__ __half sP[BN][BM];           // transposed: rows contiguous per key

    const int tid = threadIdx.x;
    const int hn  = blockIdx.y;                 // head*N + batch packed index
    const int q0  = blockIdx.x * BM;

    const int gid   = tid / TPG;
    const int lan   = tid % TPG;
    const int r0    = gid * R;
    const int kbase = lan * KEYS;
    const int cbase = lan * COLS;

    const int h_idx = hn % H;
    const int n_idx = hn / H;
    const float * q_base = Q + (size_t) hn * L_q * D;
    const float * k_base = K + (size_t) hn * L_k * D;
    // v is [D, H, L_k, N]; its stride between keys is D*H, between batches L_k*D*H.
    const float * v_base = V + (size_t) h_idx * D + (size_t) n_idx * L_k * D * H;

    for (int i = tid; i < BM * D; i += NTHREADS) {
        const int row = i / D;
        const int qq  = q0 + row;
        sQ[row][i % D] = (qq < L_q) ? __float2half(q_base[(size_t) qq * D + (i % D)])
                                    : __float2half(0.f);
    }

    float acc[R][COLS];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < COLS; ++c) acc[r][c] = 0.0f;

    float m[R], l[R];
    for (int r = 0; r < R; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    __syncthreads();

    for (int j0 = 0; j0 < L_k; j0 += BN) {
        for (int i = tid; i < BN * D; i += NTHREADS) {
            const int jj = j0 + i / D;
            const int dd = i % D;
            const bool ok = jj < L_k;
            sK[i / D][dd] = ok ? __float2half(k_base[(size_t) jj * D + dd]) : __float2half(0.f);
            sV[i / D][dd] = ok ? __float2half(v_base[(size_t) jj * D * H + dd]) : __float2half(0.f);
        }
        __syncthreads();

        float s[R][KEYS];
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < KEYS; ++t) s[r][t] = 0.0f;

        for (int d = 0; d < D; d += 4) {
            half4v qh[R];
            for (int r = 0; r < R; ++r)
                qh[r] = *reinterpret_cast<const half4v *>(&sQ[r0 + r][d]);
            for (int t = 0; t < KEYS; ++t) {
                const half4v kh = *reinterpret_cast<const half4v *>(&sK[kbase + t][d]);
                for (int r = 0; r < R; ++r)
                    s[r][t] = attn_v2_dot2(qh[r].a, kh.a, attn_v2_dot2(qh[r].b, kh.b, s[r][t]));
            }
        }
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < KEYS; ++t) s[r][t] *= scale;

        for (int r = 0; r < R; ++r) {
            float tmax = s[r][0];
            for (int t = 1; t < KEYS; ++t) tmax = fmaxf(tmax, s[r][t]);
            for (int off = 1; off < TPG; off <<= 1) tmax = fmaxf(tmax, __shfl_xor(tmax, off));

            const float mNew = fmaxf(m[r], tmax);
            const float sc   = __expf(m[r] - mNew);
            float tsum = 0.0f;
            for (int t = 0; t < KEYS; ++t) {
                const float e = __expf(s[r][t] - mNew);
                tsum += e;
                sP[kbase + t][r0 + r] = __float2half(e);
            }
            for (int off = 1; off < TPG; off <<= 1) tsum += __shfl_xor(tsum, off);

            l[r] = l[r] * sc + tsum;
            m[r] = mNew;
            for (int c = 0; c < COLS; ++c) acc[r][c] *= sc;
        }
        __syncthreads();

        for (int j = 0; j < BN; ++j) {
            float pv[R];
            const half4v ph = *reinterpret_cast<const half4v *>(&sP[j][r0]);
            const float2 p0 = __half22float2(ph.a);
            const float2 p1 = __half22float2(ph.b);
            pv[0] = p0.x; pv[1] = p0.y; pv[2] = p1.x; pv[3] = p1.y;
            for (int c = 0; c < COLS; c += 4) {
                const half4v vh = *reinterpret_cast<const half4v *>(&sV[j][cbase + c]);
                const float2 v0 = __half22float2(vh.a);
                const float2 v1 = __half22float2(vh.b);
                for (int r = 0; r < R; ++r) {
                    acc[r][c]     += pv[r] * v0.x;
                    acc[r][c + 1] += pv[r] * v0.y;
                    acc[r][c + 2] += pv[r] * v1.x;
                    acc[r][c + 3] += pv[r] * v1.y;
                }
            }
        }
        __syncthreads();
    }

    const int h = h_idx;
    const int n = n_idx;
    for (int r = 0; r < R; ++r) {
        const int qq = q0 + r0 + r;
        if (qq >= L_q) continue;
        const float inv = 1.0f / l[r];
        float * orow = O + ((size_t) n * L_q + qq) * (size_t) (D * H) + (size_t) h * D;
        for (int c = 0; c < COLS; ++c) orow[cbase + c] = acc[r][c] * inv;
    }
}

static void ggml_cuda_attn_v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const int64_t D   = q->ne[0];
    const int64_t H   = q->ne[2];
    const int64_t L_q = q->ne[1];
    const int64_t L_k = k->ne[1];
    const int64_t N   = q->ne[3];
    // q was reshaped to [D, L_q, H*N]; recover the batch factor
    const int64_t HN  = q->ne[2];
    const int64_t heads = k->ne[2] != 0 ? (k->ne[2]) : HN;
    (void) heads;

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    static long g_calls = 0;
    if (g_calls < 6) {
        fprintf(stderr, "[ATTN_V2] call#%ld D=%lld H=%lld L_q=%lld L_k=%lld N=%lld HN=%lld\n",
                g_calls, (long long) D, (long long) H, (long long) L_q, (long long) L_k, (long long) N, (long long) HN);
    }
    ++g_calls;
    const dim3 grid((unsigned) ((L_q + ATTN_V2_BM - 1) / ATTN_V2_BM), (unsigned) HN);
    attn_v2_kernel<ATTN_V2_BM, ATTN_V2_BN, ATTN_V2_R, ATTN_V2_THREADS>
        <<<grid, ATTN_V2_THREADS, 0, ctx.stream()>>>(
            (const float *) q->data, (const float *) k->data, (const float *) v->data,
            (float *) dst->data, (int) L_q, (int) L_k, (int) H, (int) N, (int) D, scale);
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_attn_v2_supported(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    if (q == nullptr || k == nullptr || v == nullptr) return false;
    return q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(q) && ggml_is_contiguous(k) && ggml_is_contiguous(v) &&
           q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 &&
           k->ne[1] == v->ne[2] &&                    // L_k matches
           q->ne[2] == k->ne[2] &&                    // both heads*N
           v->ne[1] * v->ne[3] == q->ne[2] &&         // H*N from the 4D v
           q->ne[1] >= 1024 && k->ne[1] >= 1024 &&   // only worth it for long sequences
           v->ne[1] >= 16;                            // and only for many heads
}
