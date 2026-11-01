# MiniMax-H3 on a single AMD RDNA2 GPU (Windows)

Deployment, profiling and optimization notes for running **MiniMax-H3** (33B omni
video+audio diffusion transformer) on an **AMD RX 6900 XT** under **Windows**, using
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) with the
ROCm/HIP backend.

Everything here was measured on one machine. The numbers are reproducible with the
included scripts; the analysis documents explain *why* they are what they are.

**Result: 3707 s → 191 s for a 480p / 2.3 s clip (19.4×)**, with a documented
breakdown of where the remaining time goes.

---

## What is in here

| Path | Contents |
| --- | --- |
| `scripts/gen.ps1` | Main generator. Presets `draft` / `fast` / `quality` / `longhd`, backend selection, automatic VRAM budgeting. |
| `scripts/serve.ps1` | Local web UI (sd-server on 127.0.0.1:1234). |
| `scripts/bench.ps1` | Benchmark harness (per-step timing, log parsing). |
| `scripts/extract_rocm.ps1` | Admin-free extraction of the ROCm runtime from AMD's HIP SDK installer. |
| `scripts/build_hip.bat` | Build stable-diffusion.cpp with the HIP backend (MSVC + clang from the ROCm SDK). |
| `docs/PITFALLS.md` | 40+ traps hit during the work: network, ROCm deployment, ggml behaviour, measurement methodology. |
| `docs/KERNEL-ANALYSIS.md` | Operator-level analysis: where the time goes, what RDNA2 can and cannot do, and the measured effect of every knob. |
| `docs/FRAME56-CLIFF.md` | Case study of a 2.8× cliff caused by WDDM silently demoting allocations to shared memory. |
| `docs/ATTENTION-KERNEL-BENCH.md` | Standalone HIP attention kernel: 8.76 TFLOPS on gfx1030 (18–19% of peak). |

Model weights are **not** included. The scripts download them from ModelScope /
Hugging Face.

---

## Requirements

- Windows 10/11
- AMD GPU with ROCm/HIP support (tested: RX 6900 XT, gfx1030, 16 GB)
- 64 GB system RAM recommended (mmap + offload keeps RSS around 23–28 GB)
- ~31 GB disk for the models

## Quick start

```powershell
# 1. Download the models (~31 GB)
.\scripts\download_models.ps1

# 2. Extract the ROCm runtime (no admin required)
.\scripts\extract_rocm.ps1

# 3. Generate
.\gen.ps1 -Prompt "<at least 25 words, details help>" -Preset fast
```

Presets:

| Preset | Resolution | Frames | Steps | Measured |
| --- | --- | --- | --- | --- |
| `fast` | 864×480 | 22 (0.9 s) | 8 | **69.8 s** |
| `draft` | 512×288 | 56 (2.3 s) | 8 | **72.2 s** |
| `quality` | 864×480 | 22 (0.9 s) | 20 | **138.5 s** |
| `longhd` | 864×480 | 56 (2.3 s) | 8 | **191.8 s** |

---

## The three findings that mattered

### 1. WDDM silently demotes large allocations to shared memory (2.8×)

On Windows, once ~11.4 GB of weights are resident, activation tensors larger than
~600 MB no longer fit the effective VRAM budget and the OS transparently pages them
to shared memory. `ggml` still sees them as device memory, so nothing in the logs
says "spilling" — but elementwise ops drop from 240 GB/s to 5–12 GB/s.

Fix: cap the staged weight VRAM so activations stay resident.

```
--max-vram ROCm0=11
```

The cap is bounded on **both** sides (measured at 56 frames): 9 GB → 147 s,
11 GB → 145 s, 12 GB → 145 s, **13 GB → 400 s**. The rule is *leave 4–5 GB for
activations*, i.e. roughly `VRAM − 5 GB` on other cards.

### 2. The text encoder was 20% of every run

The TE stage cost 18 s regardless of prompt length — pure weight staging for the
17 GB Qwen3-VL encoder. Switching to the 12.5 GB Q2_K_M quant cut it to 6.4 s
(**−15% end-to-end**), with no visible quality difference in A/B frames.

### 3. On Vulkan, `GGML_PREC_F32` disables the fast attention path (2.4×)

`stable-diffusion.cpp` forces `GGML_PREC_F32` on the attention QK^T matmul. In the
Vulkan backend, `ggml_vk_get_mul_mat_mat_f16acc()` requires `prec == GGML_PREC_DEFAULT`,
so that single line forces a pure-f32 shader GEMM — measured at **0.8 TFLOPS (1.7% of
peak)**. Turning on flash attention fixes it: 94.1 → 38.9 s/step, **2.4×**.

---

## Where the time goes now

At 22 frames / 8 steps / 864×480 (sampling loop only):

| Layer | Effective throughput | % of 46.08 TFLOPS peak |
| --- | --- | --- |
| End-to-end (incl. TE + VAE) | 14.1 TFLOPS | 30.6% |
| Sampling loop | 19.2 TFLOPS | 41.6% |
| MUL_MAT (q4_K GEMM) alone | ~22.5 TFLOPS | 49% |
| Isolated q4_K kernel | 24.2 TFLOPS | 52.5% |
| RDNA2 practical SIMT ceiling | ~30 TFLOPS | ~65% |

The GEMM is at 94% of its own kernel floor. Going meaningfully faster needs either
a better q4_K GEMM kernel or hardware with matrix cores (RDNA3+ WMMA / NVIDIA tensor
cores) — parameter tuning is exhausted, and `docs/KERNEL-ANALYSIS.md` documents the
negative results (stream-K +23%, FORCE_MMQ +45%, tile changes within noise).

---

## License

The scripts and documentation in this repository are MIT licensed.

`stable-diffusion.cpp` is MIT. MiniMax-H3 model weights are **not** included and remain
under their own license — check the model card on ModelScope/Hugging Face before use.

## Citation / thanks

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — the inference engine
- [ggml](https://github.com/ggml-org/ggml) — the tensor library
- AMD HIP SDK — the toolchain
