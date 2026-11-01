// Standalone q4_K GEMM microbenchmark for gfx1030.
//
// Drives ggml's own CUDA/ROCm backend (same kernel the product uses) so the
// number is directly comparable to in-situ performance, and config changes
// only require rebuilding ggml-hip (not sd.cpp).
//
// Shapes mimic MiniMax-H3 FFN projections:
//   w : [K, M] Q4_K      (weights)
//   x : [K, N] F32       (activations, N = token count)
//   y = mul_mat(w, x)    -> [M, N] F32
//
// FLOPs = 2 * M * N * K.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int64_t K = (argc > 1) ? atoll(argv[1]) : 7168;
    const int64_t M = (argc > 2) ? atoll(argv[2]) : 28672;
    const int64_t N = (argc > 3) ? atoll(argv[3]) : 2835;
    const int iters  = (argc > 4) ? atoi(argv[4]) : 20;
    const char* typeName = (argc > 5) ? argv[5] : "q4_K";
    ggml_type qtype = GGML_TYPE_Q4_K;
    if (!strcmp(typeName, "q2_K")) qtype = GGML_TYPE_Q2_K;
    else if (!strcmp(typeName, "q3_K")) qtype = GGML_TYPE_Q3_K;
    else if (!strcmp(typeName, "q4_K")) qtype = GGML_TYPE_Q4_K;
    else if (!strcmp(typeName, "q5_K")) qtype = GGML_TYPE_Q5_K;
    else if (!strcmp(typeName, "q6_K")) qtype = GGML_TYPE_Q6_K;
    else if (!strcmp(typeName, "q8_0")) qtype = GGML_TYPE_Q8_0;
    else if (!strcmp(typeName, "q4_0")) qtype = GGML_TYPE_Q4_0;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("failed to init cuda/rocm backend\n");
        return 1;
    }

    const size_t ctx_size = 64u * 1024u * 1024u;
    // no_alloc=true: tensor data lives in the backend (VRAM) buffer allocated
    // below, not in this host-side context pool.
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc*/ true };
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* w = ggml_new_tensor_2d(ctx, qtype, K, M);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, N);
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("failed to allocate tensors\n");
        return 1;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> wf((size_t) K * M);
    for (auto& v : wf) v = dist(rng);
    std::vector<char> wq(ggml_nbytes(w));
    const size_t written = ggml_quantize_chunk(qtype, wf.data(), wq.data(),
                                               0, M, K, nullptr);
    if (written != ggml_nbytes(w)) {
        printf("quantize size mismatch: %zu vs %zu\n", written, ggml_nbytes(w));
    }
    ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));

    std::vector<float> xf((size_t) K * N);
    for (auto& v : xf) v = dist(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double t1 = now_ms();

    const double ms_per   = (t1 - t0) / iters;
    const double flops    = 2.0 * (double) M * (double) N * (double) K;
    const double tflops   = flops / (ms_per * 1e9);
    const double gb_weights = (double) ggml_nbytes(w) / 1e9;

    printf("type=%s K=%lld M=%lld N=%lld  iters=%d\n", typeName, (long long) K, (long long) M, (long long) N, iters);
    printf("  %.2f ms/call   %.2f TFLOPS   weights=%.2f GB  (%.0f GB/s over %.2f ms)\n",
           ms_per, tflops, gb_weights, gb_weights / (ms_per * 1e-3), ms_per);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
