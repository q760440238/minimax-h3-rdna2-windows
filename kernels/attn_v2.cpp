// Attention kernel v2: 2D register blocking (4 query rows per thread) to cut
// LDS traffic per FMA. Target: > 8 TFLOPS on gfx1030 for H=56, D=128.
//
// Layout: 128 threads, BM=32 queries/block, BN=32 keys/tile.
//   row groups : 8 groups of 4 rows      -> gid = tid / 16
//   lane in grp: 16 threads              -> lan = tid % 16
//   QK  : thread owns 4 rows x 2 keys    -> 8 fp32 accumulators
//   PV  : thread owns 4 rows x 8 cols    -> 32 fp32 accumulators
//   LDS : sQ 8KB + sK 8KB + sV 8KB + sP 4KB = 28KB  (2 workgroups / CU)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                          \
    do {                                                                         \
        hipError_t _e = (expr);                                                  \
        if (_e != hipSuccess) {                                                  \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
                        __FILE__, __LINE__);                                     \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct alignas(8) half4v { __half2 a, b; };

// v_dot2c_f32_f16: two fp16 products accumulated into an fp32 register.
__device__ __forceinline__ float dot2(__half2 a, __half2 b, float c) {
#if defined(__AMDGCN__) && (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
                           defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx90a__))
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    const float2 af = __half22float2(a);
    const float2 bf = __half22float2(b);
    return c + af.x * bf.x + af.y * bf.y;
#endif
}

static constexpr int D  = 128;
static constexpr int THREADS = 128;

template <int BM, int BN, int ROWS_PER_THREAD, int NTHREADS = 128>
__global__ void attn_v2(const __half* __restrict__ Q,

                        const __half* __restrict__ K,
                        const __half* __restrict__ V,
                        float* __restrict__ O,
                        int N, float scale) {
    constexpr int ROW_GROUPS = BM / ROWS_PER_THREAD;
    constexpr int TPG = NTHREADS / ROW_GROUPS;
    constexpr int KEYS_PER_THREAD = BN / TPG;
    constexpr int COLS_PER_THREAD = D / TPG;
    __shared__ __half sQ[BM][D];
    __shared__ __half sK[BN][D];
    __shared__ __half sV[BN][D];
    __shared__ __half sP[BN][BM];   // transposed: rows contiguous per key

    const int tid = threadIdx.x;
    const int h   = blockIdx.y;
    const int q0  = blockIdx.x * BM;

    const int gid   = tid / TPG;
    const int lan   = tid % TPG;
    const int r0    = gid * ROWS_PER_THREAD;
    const int kbase = lan * KEYS_PER_THREAD;
    const int cbase = lan * COLS_PER_THREAD;

    for (int i = tid; i < BM * D; i += NTHREADS)
        sQ[i / D][i % D] = Q[((size_t)h * N + q0 + i / D) * D + (i % D)];

    float acc[ROWS_PER_THREAD][COLS_PER_THREAD];
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
        for (int c = 0; c < COLS_PER_THREAD; ++c) acc[r][c] = 0.0f;

    float m[ROWS_PER_THREAD], l[ROWS_PER_THREAD];
    for (int r = 0; r < ROWS_PER_THREAD; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    __syncthreads();

    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        for (int i = tid; i < BN * D; i += NTHREADS) {
            const int jj = j0 + i / D;
            const int dd = i % D;
            const bool ok = jj < N;
            sK[i / D][dd] = ok ? K[((size_t)h * N + jj) * D + dd] : __float2half(0.f);
            sV[i / D][dd] = ok ? V[((size_t)h * N + jj) * D + dd] : __float2half(0.f);
        }
        __syncthreads();

        // ---- S = Q(4 rows) . K(2 keys) ----
        float s[ROWS_PER_THREAD][KEYS_PER_THREAD];
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
            for (int t = 0; t < KEYS_PER_THREAD; ++t) s[r][t] = 0.0f;

        for (int d = 0; d < D; d += 4) {
            half4v qh4[ROWS_PER_THREAD];
            for (int r = 0; r < ROWS_PER_THREAD; ++r)
                qh4[r] = *reinterpret_cast<const half4v*>(&sQ[r0 + r][d]);
            for (int t = 0; t < KEYS_PER_THREAD; ++t) {
                const half4v kh = *reinterpret_cast<const half4v*>(&sK[kbase + t][d]);
                for (int r = 0; r < ROWS_PER_THREAD; ++r)
                    s[r][t] = dot2(qh4[r].a, kh.a, dot2(qh4[r].b, kh.b, s[r][t]));
            }
        }
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
            for (int t = 0; t < KEYS_PER_THREAD; ++t) s[r][t] *= scale;

        // ---- online softmax; TPG threads (one per key chunk) cooperate per row ----
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            float tmax = s[r][0];
            for (int t = 1; t < KEYS_PER_THREAD; ++t) tmax = fmaxf(tmax, s[r][t]);
            for (int off = 1; off < TPG; off <<= 1) tmax = fmaxf(tmax, __shfl_xor(tmax, off));

            const float mNew = fmaxf(m[r], tmax);
            const float sc   = __expf(m[r] - mNew);
            float tsum = 0.0f;
            for (int t = 0; t < KEYS_PER_THREAD; ++t) {
                const float e = __expf(s[r][t] - mNew);
                tsum += e;
                sP[kbase + t][r0 + r] = __float2half(e);
            }
            for (int off = 1; off < TPG; off <<= 1) tsum += __shfl_xor(tsum, off);

            l[r] = l[r] * sc + tsum;
            m[r] = mNew;
            for (int c = 0; c < COLS_PER_THREAD; ++c) acc[r][c] *= sc;
        }
        __syncthreads();

        // ---- O(4 rows x 8 cols) += P(4x32) @ V(32x8) ----
        for (int j = 0; j < BN; ++j) {
            float pv[ROWS_PER_THREAD];
            const half4v ph = *reinterpret_cast<const half4v*>(&sP[j][r0]);
            const float2 p0 = __half22float2(ph.a);
            const float2 p1 = __half22float2(ph.b);
            pv[0] = p0.x; pv[1] = p0.y; pv[2] = p1.x; pv[3] = p1.y;
            float4 vv[COLS_PER_THREAD / 4];
            for (int c = 0; c < COLS_PER_THREAD; c += 4) {
                const half4v vh = *reinterpret_cast<const half4v*>(&sV[j][cbase + c]);
                const float2 v0 = __half22float2(vh.a);
                const float2 v1 = __half22float2(vh.b);
                vv[c / 4] = make_float4(v0.x, v0.y, v1.x, v1.y);
            }
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                for (int c = 0; c < COLS_PER_THREAD; c += 4) {
                    acc[r][c]     += pv[r] * vv[c / 4].x;
                    acc[r][c + 1] += pv[r] * vv[c / 4].y;
                    acc[r][c + 2] += pv[r] * vv[c / 4].z;
                    acc[r][c + 3] += pv[r] * vv[c / 4].w;
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        float* ov = O + ((size_t)h * N + q0 + r0 + r) * D;
        const float inv = 1.0f / l[r];
        for (int c = 0; c < COLS_PER_THREAD; ++c) ov[cbase + c] = acc[r][c] * inv;
    }
}

template <int BM, int BN, int R, int NT = 128>
static float run_v2(const __half* dQ, const __half* dK, const __half* dV, float* dO,
                    int H, int N, int iters) {
    const dim3 grid((N + BM - 1) / BM, H);
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    attn_v2<BM, BN, R, NT><<<grid, NT>>>(dQ, dK, dV, dO, N, 1.0f / std::sqrt((float)D));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i)
        attn_v2<BM, BN, R, NT><<<grid, NT>>>(dQ, dK, dV, dO, N, 1.0f / std::sqrt((float)D));
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a);
    hipEventDestroy(b);
    return ms / iters;
}

// CPU reference for small N (validates the kernel, not the benchmark).
static void cpu_attn(const std::vector<__half>& Q, const std::vector<__half>& K,
                     const std::vector<__half>& V, std::vector<float>& O,
                     int H, int N) {
    const float scale = 1.0f / std::sqrt((float)D);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < N; ++i) {
            std::vector<float> s(N);
            float mx = -INFINITY;
            for (int j = 0; j < N; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += __half2float(Q[((size_t)h * N + i) * D + d]) *
                           __half2float(K[((size_t)h * N + j) * D + d]);
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            for (int d = 0; d < D; ++d) {
                float a = 0.0f;
                for (int j = 0; j < N; ++j)
                    a += s[j] * __half2float(V[((size_t)h * N + j) * D + d]);
                O[((size_t)h * N + i) * D + d] = a / sum;
            }
        }
    }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 2430;
    const int H = 56;
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 10;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s)  N=%d H=%d D=%d iters=%d\n",
                prop.name, prop.gcnArchName, N, H, D, iters);

    const size_t n = (size_t)H * N * D;
    const size_t bytes = n * sizeof(__half);

    __half *dQ, *dK, *dV;
    float* dO;
    HIP_CHECK(hipMalloc(&dQ, bytes));
    HIP_CHECK(hipMalloc(&dK, bytes));
    HIP_CHECK(hipMalloc(&dV, bytes));
    HIP_CHECK(hipMalloc(&dO, n * sizeof(float)));

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<__half> hQ(n), hK(n), hV(n);
    for (size_t i = 0; i < n; ++i) {
        hQ[i] = __float2half(dist(rng));
        hK[i] = __float2half(dist(rng));
        hV[i] = __float2half(dist(rng));
    }
    HIP_CHECK(hipMemcpy(dQ, hQ.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, hK.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, hV.data(), bytes, hipMemcpyHostToDevice));

    // correctness on a small case
    {
        const int Nc = 128, Hc = 2;
        const size_t nc = (size_t)Hc * Nc * D;
        __half *cq, *ck, *cv;
        float* co;
        HIP_CHECK(hipMalloc(&cq, nc * sizeof(__half)));
        HIP_CHECK(hipMalloc(&ck, nc * sizeof(__half)));
        HIP_CHECK(hipMalloc(&cv, nc * sizeof(__half)));
        HIP_CHECK(hipMalloc(&co, nc * sizeof(float)));
        HIP_CHECK(hipMemcpy(cq, hQ.data(), nc * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(ck, hK.data(), nc * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cv, hV.data(), nc * sizeof(__half), hipMemcpyHostToDevice));
        attn_v2<32, 32, 4><<<dim3(Nc / 32, Hc), THREADS>>>(cq, ck, cv, co, Nc, 1.0f / std::sqrt((float)D));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> gpu(nc), ref(nc);
        HIP_CHECK(hipMemcpy(gpu.data(), co, nc * sizeof(float), hipMemcpyDeviceToHost));
        cpu_attn(hQ, hK, hV, ref, Hc, Nc);
        double maxErr = 0.0, sum = 0.0;
        for (size_t i = 0; i < nc; ++i) {
            maxErr = std::max(maxErr, std::fabs((double)gpu[i] - (double)ref[i]));
            sum += std::fabs((double)ref[i]);
        }
        std::printf("correctness N=%d H=%d: max_abs_err=%.6f  mean|ref|=%.6f\n",
                    Nc, Hc, maxErr, sum / nc);
        HIP_CHECK(hipFree(cq)); HIP_CHECK(hipFree(ck));
        HIP_CHECK(hipFree(cv)); HIP_CHECK(hipFree(co));
    }

    const double flops = 4.0 * H * (double)N * (double)N * D;
    float m1 = run_v2<32, 32, 4>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM32 BN32 R4 : %8.2f ms  %6.2f TFLOPS\n", m1, flops / (m1 * 1e9));
    float m2 = run_v2<32, 64, 4>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM32 BN64 R4 : %8.2f ms  %6.2f TFLOPS\n", m2, flops / (m2 * 1e9));
    float m3 = run_v2<16, 64, 4>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM16 BN64 R4 : %8.2f ms  %6.2f TFLOPS\n", m3, flops / (m3 * 1e9));
    float m4 = run_v2<64, 32, 8>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM64 BN32 R8 : %8.2f ms  %6.2f TFLOPS\n", m4, flops / (m4 * 1e9));
    float m5 = run_v2<64, 64, 8>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM64 BN64 R8 : %8.2f ms  %6.2f TFLOPS\n", m5, flops / (m5 * 1e9));
    float m6 = run_v2<32, 32, 8>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM32 BN32 R8 : %8.2f ms  %6.2f TFLOPS\n", m6, flops / (m6 * 1e9));
    float m7 = run_v2<64, 32, 4>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM64 BN32 R4 : %8.2f ms  %6.2f TFLOPS\n", m7, flops / (m7 * 1e9));
    float m8 = run_v2<64, 32, 4, 256>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM64 BN32 R4 T256 : %8.2f ms  %6.2f TFLOPS\n", m8, flops / (m8 * 1e9));
    float m9 = run_v2<128, 32, 4, 256>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM128 BN32 R4 T256 : %8.2f ms  %6.2f TFLOPS\n", m9, flops / (m9 * 1e9));

    float m11 = run_v2<64, 32, 2>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM64 BN32 R2 : %8.2f ms  %6.2f TFLOPS\n", m11, flops / (m11 * 1e9));
    float m12 = run_v2<128, 32, 8, 256>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM128 BN32 R8 T256 : %8.2f ms  %6.2f TFLOPS\n", m12, flops / (m12 * 1e9));
    float m13 = run_v2<128, 32, 4, 512>(dQ, dK, dV, dO, H, N, iters);
    std::printf("v2 BM128 BN32 R4 T512 : %8.2f ms  %6.2f TFLOPS\n", m13, flops / (m13 * 1e9));


    HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dK));
    HIP_CHECK(hipFree(dV)); HIP_CHECK(hipFree(dO));
    return 0;
}
