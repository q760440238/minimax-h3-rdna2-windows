# ATTENTION-V2: integration and measurement

- Machine: RX 6900 XT (gfx1030), ROCm 7.2, Windows, `sd-cli` from `build-hip`
  (configure already done, rebuilt with `E:\minimaxh3\build_hip.bat build`).
- Tree: `E:\minimaxh3\src\sdcpp\stable-diffusion.cpp-37bc88b5e97a213821949732ea66c58ae8ffc784`.
- Model: `minimax_h3_fl2va_pruned-Q4_K.gguf`, qwen3vl-32B Q2_K text encoder.
- Env: `SD_ATTN_V2=1` opts into the kernel (default off), `SD_ATTN_PROBE=1` turns on the
  shape probe described below (default off).

## TL;DR

**Verdict: keep the kernel opt-in (`SD_ATTN_V2=1`); do not enable it by default.**
The integration is fixed — the op now reaches the model's real self-attention
(H=56, D=128, L_q=L_k=7132 at 56 frames, N=1) and runs without errors — but at 56 frames
/ 8 steps it is **22% slower** than the default flash-attention path:
**447.8 s (56.0 s/step) vs 366.2 s (45.8 s/step)**. That is one clean run per arm (n=1+1);
the second off/on pair was destroyed by external GPU contention (§6). The default path
(`SD_ATTN_V2` unset) is unchanged, and the tiny-attention ("H=4, L=1024", 20x slowdown)
problem is excluded by the new gate.

## 1. Probe instrumentation

`ggml_ext_attention_ext()` in `src/core/ggml_extend.cpp` now has an env-gated probe
(`SD_ATTN_PROBE=1`) that prints, for the first 15 attention calls **and** the first 15
"large MHA candidates" (`mask == nullptr && d_head == 128 && n_kv_head == n_head &&
L_q >= 1024 && L_k >= 1024`):

- `d_head`, `n_head`, `n_kv_head`, `L_q`, `L_k`, `N`, `skip_reshape`, `flash`, `mask != null`
- type, `ne[]`, `nb[]` and `ggml_is_contiguous()` for q, k, v **as received**
- the same shapes **after** the function's own reshape/permute
- the gate values of the ATTN_V2 predicate and the path actually chosen
  (`flash` / `attn_v2` / `sage` / `manual`)

The first 15 calls alone are useless for this model: calls 0-4 are the startup
flash-attention auto-probe (`src/core/probe/flash_attn.cpp`) and calls 5-54 are the
qwen3vl text encoder. The DiT self-attention is call #57, hence the second filter.

## 2. Probe log before the fix

`out/bench_attnprobe2.log`, 2 steps / 22 frames, `SD_ATTN_V2=1 SD_ATTN_PROBE=1`
(line breaks inserted by the PowerShell redirect have been joined):

```
[ATTN_V2_PROBE #3] entry skip_reshape=0 big=1 flash=0 sage=0 mask=0 d_head=128 n_head=4 n_kv_head=4 L_q=1024 L_k=1024 N=1 | q type=f32 ne=[512,1024,1,1] nb=[2048,2097152,2097152] cont=1 | k type=f32 ne=[512,1024,1,1] nb=[2048,2097152,2097152] cont=1 | v type=f32 ne=[512,1024,1,1] nb=[2048,2097152,2097152] cont=1
[ATTN_V2_PROBE #3] gates env=1 kqvfree=1 masknull=1 d128=1 mha=1 f32=1 qcont=1 kcont=1 vcont=1 size=0 | post q ne=[128,1024,4,1] | k ne=[128,1024,4,1] | v ne=[128,4,1024,1] nb=[512,2048,2097152]
[ATTN_V2_PROBE #3] out path=manual kqv ne=[128,4,1024,1] type=f32
[ATTN_V2_PROBE #5] entry skip_reshape=1 big=0 flash=1 sage=0 mask=1 d_head=128 n_head=64 n_kv_head=8 L_q=61 L_k=61 N=1 | q type=f32 ne=[128,61,64,1] nb=[512,31232,1998848] cont=1 | k type=f32 ne=[128,61,8,1] nb=[512,31232,249856] cont=1 | v type=f32 ne=[128,8,61,1] nb=[512,4096,249856] cont=1
[ATTN_V2_PROBE #5] gates env=1 kqvfree=0 masknull=0 d128=1 mha=0 f32=1 qcont=1 kcont=1 vcont=1 size=0 | post q ne=[128,61,64,1] | k ne=[128,61,8,1] | v ne=[128,8,61,1] nb=[512,4096,249856]
[ATTN_V2_PROBE #5] out path=flash kqv ne=[128,64,61,1] type=f32
[ATTN_V2_PROBE #57] entry skip_reshape=1 big=1 flash=1 sage=0 mask=0 d_head=128 n_head=56 n_kv_head=56 L_q=2970 L_k=2970 N=1 | q type=f32 ne=[128,2970,56,1] nb=[512,1520640,85155840] cont=1 | k type=f32 ne=[128,2970,56,1] nb=[512,1520640,85155840] cont=1 | v type=f32 ne=[128,56,2970,1] nb=[512,86016,255467520] cont=0
[ATTN_V2_PROBE #57] gates env=1 kqvfree=0 masknull=1 d128=1 mha=1 f32=1 qcont=1 kcont=1 vcont=0 size=1 | post q ne=[128,2970,56,1] | k ne=[128,2970,56,1] | v ne=[128,56,2970,1] nb=[512,86016,255467520]
[ATTN_V2_PROBE #57] out path=flash kqv ne=[128,56,2970,1] type=f32
```

## 3. Diagnosis

The model's main self-attention is reachable, but **two independent gates** stopped it:

1. **`v` is not contiguous.** `minimax_h3.hpp::attend()` builds q/k/v as views of one
   fused `qkv_proj` output `[3C, L, B]` and passes `skip_reshape=true`. q and k are
   re-laid out by `attention_layout()` (`cont(permute(...))`), so they arrive contiguous:
   `q ne=[128,2970,56,1] nb=[512,1520640,85155840]`, which is exactly contiguous
   (512 = 4·128, 1520640 = 512·2970, 85155840 = 1520640·56). v is left as the raw view
   `ne=[128,56,2970,1] nb=[512,86016,255467520]`: the per-token stride is
   86016 = 3 × 28672 (3·C·4), i.e. v is every third 128-float block of the fused qkv
   buffer. `ggml_is_contiguous(v) == 0`, so the old predicate
   (`... && ggml_is_contiguous(q) && ggml_is_contiguous(k) && ggml_is_contiguous(v)`)
   rejected it. This was the primary blocker.
2. **The flash-attention branch ran first.** The ROCm backend auto-enables flash
   attention at startup (`flash attention auto: ROCm0 -> on (probe: normal 0.680 ms,
   flash 0.435 ms)`), so the `if (kqv == nullptr && flash_attn)` block above the
   ATTN_V2 block already set `kqv` for every DiT layer (`kqvfree=0` in the gate line).
   Even a contiguous v would never have reached the op.

Two things that were **not** the problem:

- **Mask**: `mask=0` (null) for the DiT self-attention.
- **Types / heads**: all F32, `n_head == n_kv_head == 56`, `d_head == 128`, `N == 1`.

About the earlier "H=4, L=1024" observation: that is the startup flash auto-probe
(`src/core/probe/flash_attn.cpp` builds `heads=4, head_size=128, length=1024`), not model
attention. The historical kernel line `D=128 H=4 L_q=1024 L_k=1024 N=1 HN=4
q_ne=128,1024,4,1` matches that probe graph exactly. With the current predicate
(`v->ne[1] >= 16`) it is already rejected (`size=0` above), and the pre-fix probe run
completed at normal speed (17-20 s for 2 steps); the reported 103 s/step did not
reproduce with this tree. The qwen3vl text encoder (`n_head=64`, `n_kv_head=8`, mask
present, L=61) is correctly rejected as well.

## 4. Predicate change

In `src/core/ggml_extend.cpp` the ATTN_V2 attempt was moved **above** the flash-attention
block and its gate rewritten. Non-contiguous q/k/v are materialized with
`ggml_ext_cont()` only inside the gated block:

```cpp
    static const bool attn_v2_enabled = [] {
        const char * e = getenv("SD_ATTN_V2");
        return e != nullptr && atoi(e) != 0;
    }();
    const bool attn_v2_shapes =
        kqv == nullptr && mask == nullptr && d_head == 128 && n_kv_head == n_head && n_head >= 16 &&
        L_q >= 1024 && L_k >= 1024 && N == 1 &&
        q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 &&
        q->ne[0] == d_head && k->ne[0] == d_head && v->ne[0] == d_head &&
        q->ne[2] == n_head && k->ne[2] == n_kv_head &&
        v->ne[1] == n_head && v->ne[2] == L_k && v->ne[3] == 1;
    if (attn_v2_enabled && attn_v2_shapes) {
        auto q_in    = ggml_ext_cont(ctx, q);
        auto k_in    = ggml_ext_cont(ctx, k);
        auto v_in    = ggml_ext_cont(ctx, v);
        auto attn_v2 = ggml_attn_v2(ctx, q_in, k_in, v_in, scale);
        if (ggml_backend_supports_op(backend, attn_v2)) {
            kqv = attn_v2;   // [d_head, n_head, L_q, N]
        }
    }
```

- **Size gate** (`L_q, L_k >= 1024`, `n_head >= 16`, `d_head == 128`): the kernel is only
  worth its fixed overhead on the long DiT self-attention; VAE/audio attentions
  (`n_head=1` or 4) keep their original path.
- **Contiguity**: `ggml_ext_cont()` is a no-op for q/k here and one linear copy for v
  (`L·H·D·4 B` = 85 MB at 22 frames, ~200 MB at 56 frames, per layer; ~10 GB/step of
  copy traffic across 50 layers, i.e. tens of ms).
- **`N == 1`**: the op builds its result as `[D, q->ne[2], q->ne[1], q->ne[3]]` and the
  kernel derives `H = q->ne[2]`, `N = q->ne[3]`. For `skip_reshape` inputs q is
  `[D, L, H*N, 1]`, so the packing only round-trips when `N == 1` (always true for this
  CLI path). Rather than change the already-validated kernel, the gate enforces it, and
  `ggml_cuda_attn_v2_supported()` in `ggml/src/ggml-cuda/attn-v2.cuh` now also requires
  `q->ne[3] == 1 && v->ne[3] == 1`.
- The support predicate in `attn-v2.cuh` still enforces F32, contiguity, `D == 128`,
  `L >= 1024`, `H >= 16`, as a second line of defense.

## 5. Verification (2 steps, 22 frames)

`out/bench_verify22.log`, `SD_ATTN_V2=1 SD_ATTN_PROBE=1`:

```
[ATTN_V2_PROBE #57] gates env=1 prefree=1 masknull=1 d128=1 mha=1 f32=1 qcont=1 kcont=1 vcont=0 size=1 shapes=1 | post q ne=[128,2970,56,1] | k ne=[128,2970,56,1] | v ne=[128,56,2970,1] nb=[512,86016,255467520]
[ATTN_V2_PROBE #57] out path=attn_v2 kqv ne=[128,56,2970,1] type=f32
[ATTN_V2] call#0 D=128 H=56 L_q=2970 L_k=2970 N=1 HN=56
[ATTN_V2] call#1 D=128 H=56 L_q=2970 L_k=2970 N=1 HN=56
...
[I] sampling completed, taking 19.55s --- video.cpp:1776
[I] generate_video completed in 36.43s --- video.cpp:2008
```

The op now fires for every DiT layer with the real shapes (H=56, D=128, L_q=L_k=2970 at
22 frames = 2835 video + 135 audio tokens), no `[E]` lines, output video written
(`out/bench_verify22.webm`). At 56 frames the A/B "on" run logs the larger real shape:

```
[ATTN_V2] call#0 D=128 H=56 L_q=7132 L_k=7132 N=1 HN=56
```

(7132 = 6885 video + 247 audio tokens). Structural mapping was checked against the
kernel: q `[D,L,H]` head-major, v `[D,H,L,1]`, output `[D,H,L,1]`, which is exactly what
`ggml_attn_v2()` and `attn_v2_kernel` index.

## 6. A/B at 56 frames, 8 steps

Command per run (same session, alternating `SD_ATTN_V2`):

```
powershell -File bench.ps1 -Name ab56_<x> -Steps 8 -Frames 56 \
  -Sd build-hip\bin\sd-cli.exe -DeviceName ROCm0 -Llm <qwen3vl_32b Q2_K_M> \
  -Extra "--mmap --vae-tiling"
```

**Deviation from the requested command**: the requested `-Extra "--mmap --vae-tiling
--max-vram ROCm0=11"` cannot run at 56 frames — the default path aborts during weight
preparation:

```
[W] model manager cannot make enough memory available on ROCm0: need 12554.75 MB device / 12044.00 MB budget, available 15689.93 MB device / 11051.20 MB budget
[E] minimax_h3 segment 4/54 (minimax_h3.blocks.0) failed during weight preparation
```

so all four runs below used `-Extra "--mmap --vae-tiling"` (bench.ps1 default 15 GB
`--max-vram`). All runs used the same flags, same binary, same session.

| run | `SD_ATTN_V2` | sampling (8 steps) | s/step (steady) | DiT compute buffer / plan | usable? |
|---|---|---|---|---|---|
| `ab56_off1` | 0 (default) | **366.22 s** | 45.2 | 1744 MB / 1 segment | **yes** |
| `ab56_on1`  | 1 | **447.76 s** | 55.2 | 1741 MB / 1 segment | **yes** |
| `ab56_off2` | 0 | FAILED after ~585 s (steps 112-163 s/it) | — | **11896 MB / 54 segments** (anomalous plan) | no |
| `ab56_on2`  | 1 | 963.84 s | 120.5 | 1741 MB / 1 segment | no (external GPU contention) |

- Valid pair: `ab56_off1` → `ab56_on1`: **+81.5 s (+22.3%)** for the whole 8-step run,
  **+10.0 s/step**. Per-step traces are flat (off1 44.96-48.72 s, on1 54.65-60.43 s), so
  the difference is not a warm-up artifact. `ab56_off1` ran first (its startup flash probe
  was still cold: 24.0 ms vs 0.7 ms in later runs), i.e. the bias, if any, was *against*
  the baseline, yet the baseline still won.
- `ab56_off2` picked a completely different memory plan (11.9 GB compute buffer,
  54 segments instead of 1) and failed at step 5; `ab56_on2` used the same plan as
  `ab56_on1` but every step was ~120 s, consistent with another GPU workload running
  concurrently (a GEMM sweep on the same device). Both are excluded.
- Caveat stated explicitly: the decision rests on **one clean run per arm (n=1+1)**.
  The direction is unambiguous (22%, far outside the 2-3% run-to-run spread of the flat
  traces) and matches the 22-frame spot checks: 2-step 22-frame runs took 17.0 s / 20.3 s
  without the op and 19.6 s with it (too short to be a controlled A/B, but same sign).
  A proper 22-frame 8-step A/B could not be run because the GPU window was handed off.

Why the op loses here: the default path already uses flash attention (`flash attention
auto: ROCm0 -> on`), and the only graph difference between the arms is the attention op
(compute buffer 1741 vs 1744 MB, so no memory/staging effect). Attention at 56 frames is
`4·H·L²·D·50 ≈ 73 TFLOP/step`; a +10 s/step delta implies the in-tree ATTN_V2 kernel runs
at roughly **4.5-5 TFLOPS** while the F16 flash tile kernel is at ~15-20 TFLOPS. That is
well below the 8.5-8.8 TFLOPS the same kernel reaches standalone (see
`ATTENTION-KERNEL-BENCH.md`); the in-tree version additionally pays the strided-v
materialization (10 GB/step of copies across 50 layers) and a cold L2 for each layer.

## 7. Default path safety

`SD_ATTN_V2` unset (the default) means the ATTN_V2 block is skipped at the `&&` — no
`ggml_attn_v2`, no `ggml_cont` and no other graph node is created. The only added work is
one `getenv()` (once per process, `static`) and a handful of host-side shape/stride
comparisons. `SD_ATTN_PROBE` is also off by default. The default graph is therefore
identical to the pre-change graph.

This is confirmed empirically by the A/B baseline itself: `ab56_off1` (the default arm)
ran at 366.2 s / 8 steps at 56 frames with the standard 1744 MB / 1-segment plan, and the
22-frame default runs used for the probe were 17-20 s for 2 steps — no regression versus
the earlier default-path measurements in this tree. The default arm never invokes the new
code.

(The one behaviour change is inside the opt-in path only: it now also takes over from
flash attention, because the flash block used to run first. With the env var set, that is
the intended semantics.)

## 8. Recommendation

**Keep it opt-in; do not enable by default.** At 56 frames the kernel is 22% slower than
the default flash-attention path (447.8 s vs 366.2 s for 8 steps, 1 clean run per arm), so
`SD_ATTN_V2` must stay an experiment flag. **Do not revert** the integration, because it
is now correct and correctly gated, which was the actual deliverable:

- it reaches the real main attention (H=56, D=128, L=7132, N=1, mask-free, F32) instead of
  only the startup flash probe, and it runs with no errors and a plausible output;
- tiny/VAE attentions are excluded by `n_head >= 16 && L >= 1024`, so the old
  "small attention makes the step 20x slower" failure cannot come back; the H=4/L=1024
  case was the flash auto-probe, not model attention;
- the strided `v` is materialized (`ggml_ext_cont`) so the op no longer silently depends on
  the caller's layout, and `N == 1` is enforced in both the call site and the backend
  support predicate.

If the kernel is to be pursued further, the next steps are: (a) a controlled 22-frame
8-step A/B (it should lose by less there, since attention is a smaller share of the step);
(b) an A/B against `--flash-attn off` (the manual F32 path, ~2-3 TFLOPS), where this op is
expected to win; (c) profile why the in-tree kernel gets ~4.5-5 TFLOPS versus the 8.76
TFLOPS standalone (L2 behaviour per layer, the per-layer v copy, launch shape). Until then
the recommendation is: **opt-in only, default unchanged**.

## Appendix: files changed

- `src/core/ggml_extend.cpp` — probe instrumentation (`SD_ATTN_PROBE`), ATTN_V2 gate and
  placement (before flash), `ggml_ext_cont()` materialization.
- `ggml/src/ggml-cuda/attn-v2.cuh` — support predicate additionally requires
  `q->ne[3] == 1 && v->ne[3] == 1` (the only packing the kernel implements).

Not changed: `gen.ps1`, `bench.ps1`, anything outside this source tree, `E:\work-gemm`.
