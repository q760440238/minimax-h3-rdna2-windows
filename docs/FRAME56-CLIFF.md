# 56 帧性能断崖：定位报告

日期：2026-11-01
方法：在 ggml-cuda 里临时插入逐算子 HIP event 计时 + 输出字节统计（`GGML_PROFILE_OPS`），
关闭 HIP 图捕获以便计时，跑 2 步取均值。

## 现象

| 帧数 | latent T | tokens | 单步 | 每 token |
| --- | --- | --- | --- | --- |
| 22 | 7 | 2 835 | 6.8 s | 2.40 ms |
| 39 | 12 | 4 860 | 11.5 s | 2.30 ms |
| 56 | 17 | 6 885 | 50.3 s | **7.07 ms** |

22→39 是完美线性；39→56 每 token 暴涨 **3.07 倍**。

## 排除项（都实测过）

| 假设 | 实验 | 结果 |
| --- | --- | --- |
| 注意力 f32 路径 | 56 帧 FA 开 vs 关 | 385.8 s vs 389.8 s（差 1%），**排除** |
| kernel 派发切换 | 插桩统计 mmvf/mmf/mmvq/mmq/cuBLAS 调用数 | 22/39/56 三档**完全相同**（699/0/0/2174/535），**排除** |
| 显存溢出到 CPU 后端 | 看 compute buffer 分配 | 39f 1 232 MB、56f 1 744 MB，都在 VRAM；CPU 侧只有 3–4 MB，**排除** |
| VAE / 音频 | 日志 | 不计入 sampling，**排除** |

## 逐算子剖析（2 步累计）

| 算子 | 39 帧 | 56 帧 | 每次调用 | 带宽 39f → 56f |
| --- | --- | --- | --- | --- |
| MUL_MAT | 22 784 ms (71%) | 41 246 ms (41%) | 4.7 → 6.9 ms | 正常（×1.47，与 token 数 ×1.42 一致）|
| FLASH_ATTN_EXT | 5 947 ms (19%) | 17 921 ms (18%) | 8.0 → 17.3 ms | 8.0 → **1.87 GB/s** |
| SCALE | 434 ms | **18 238 ms (18%)** | 0.18 → **7.6 ms** | 287 → **未计入（异常）** |
| CONCAT | 329 ms | **9 122 ms (9%)** | 0.058 → **1.43 ms** | 234 → **12.3 GB/s** |
| GLU | 212 ms | **7 239 ms (7%)** | 0.70 → **23.8 ms** | 137 → **5.3 GB/s** |
| ADD | 440 ms | 4 926 ms (5%) | 0.11 → 0.92 ms | 308 → **49.8 GB/s** |
| CONT | 916 ms | 1 474 ms | 0.066 → 0.086 ms | 131 → 124 GB/s（正常）|
| MUL | 127 ms | 186 ms | — | 286 → 295 GB/s（正常）|

**关键对比**：同样是访存型算子，`MUL`/`CONT` 在 56 帧仍有 124–295 GB/s（正常），
而 `GLU` 掉到 5.3 GB/s、`CONCAT` 12.3 GB/s、`FLASH_ATTN` 1.87 GB/s——
这是 **host 内存 / PCIe 量级**，不是 GPU 显存带宽。

**矩阵乘本身完全正常**（每次调用 ×1.47，与 token 增长一致），说明不是算力问题。

## 结论

56 帧的 3 倍退化**不是**注意力、不是 kernel 派发、不是显存不足（按分配日志），
而是**部分访存型算子（GLU / CONCAT / SCALE / ADD / FLASH_ATTN）的带宽塌陷到 PCIe 量级**，
而同一时刻另一些算子（MUL / CONT / 矩阵乘）带宽正常。

最可能的解释（尚未证实）：这些算子的输入/输出张量被分配在 host 可见内存（或跨 PCIe 的映射），
只有部分张量受影响。

**另一个值得查的线索**：latent 时间维 T 在 39 帧是 **12**、56 帧是 **17**，中间跨过了 **16** 这个边界。
若某个 kernel 的向量化/索引假设 T 是 16 的倍数（或 <16），T=17 就会掉进标量路径，
这能解释为什么是「39→56」这个特定区间突然塌陷，而不是随 token 数平滑变化。
建议优先核对 GLU / CONCAT / softmax 相关 kernel 对 `ne[1]`/`ne[2]` 取模或对齐的假设。
下一步需要记录每个节点的 buffer 设备归属来确认。

## 下一步（按优先级）

1. 在剖析里增加 `node->buffer` 的设备类型输出，确认慢算子的张量在哪块内存
2. 对照实验：`--mmap` 开关、`--max-vram` 显式指定、`--offload-to-cpu` 开关，
   找出能让这些算子回到 VRAM 的组合
3. 若确认是 sd.cpp 的 buffer 分配策略问题，可在 `model_loader` / `ggml_extend_backend`
   里为大张量单独申请 VRAM 缓冲（类似 `--vae-tiling` 的思路，按 T 分块处理视频 latent）

## 复现方式

```powershell
# 需要带插桩的自编译版（build-hip\bin\sd-cli.exe），且关闭 HIP 图捕获
powershell -File E:\minimaxh3\bench.ps1 -Name prof56 -Steps 2 -Frames 56 `
  -Sd E:\minimaxh3\build-hip\bin\sd-cli.exe -DeviceName ROCm0 `
  -Llm E:\minimaxh3\models\text_encoders\qwen3vl_32b_minimax_h3-Q2_K_M.gguf -Extra "--mmap"
# 日志里搜 [OPPROF]
```
