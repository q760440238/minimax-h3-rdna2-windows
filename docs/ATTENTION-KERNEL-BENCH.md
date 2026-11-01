# Attention 微基准报告（gfx1030）

独立 HIP 微基准，不接 sd.cpp，用于确认 RDNA2 上纯 FMA + LDS 分块的自写注意力能达到什么水平。

- 源码：`hip/attn_v2.cpp`（最终版），`hip/attn_bench.cpp`（第一版对照）
- 构建：`hip/build_hip_kernel.bat <src>`（ROCm SDK 的 clang，`--offload-arch=gfx1030`）
- 硬件：RX 6900 XT (gfx1030)，FP16 SIMD 峰值 46.08 TFLOPS
- 形状：H=56, D=128, N=2430（22 帧）/ N=5670（56 帧）
- FLOP 口径：`4 * H * N² * D`（QK^T + P@V 两次乘加）

## 结果（2026-11-01）

| 配置 | N=2430 | N=5670 |
|---|---|---|
| **BM128 / BN32 / R8 / T256（最终）** | **19.93 ms → 8.50 TFLOPS** | **105.20 ms → 8.76 TFLOPS** |
| BM128 / BN32 / R4 / T256 | 22.52 ms → 7.52 TFLOPS | 121.94 ms → 7.56 TFLOPS |
| BM64 / BN32 / R2 | 26.26 ms → 6.45 TFLOPS | 141.88 ms → 6.50 TFLOPS |
| BM64 / BN32 / R4 | 26.36 ms → 6.42 TFLOPS | 142.82 ms → 6.45 TFLOPS |
| BM64 / BN32 / R4 / T256 | 27.40 ms → 6.18 TFLOPS | 146.28 ms → 6.30 TFLOPS |
| BM128 / BN32 / R4 / T512 | 25.97 ms → 6.52 TFLOPS | 143.29 ms → 6.43 TFLOPS |

**验收达标**：两个形状都 ≥ 8 TFLOPS（目标的 106% / 110%），等于峰值的 18.5% / 19.0%。

精度：与 CPU 参考实现对比，`max_abs_err = 1.7e-5`（N=128, H=2）。

## 优化路径（每一步的实测）

| 步骤 | TFLOPS | 关键改动 |
|---|---|---|
| 第一版（每线程 1 行、1 键组） | 2.36 | 基础分块 + online softmax |
| 每线程 4 行（2D 寄存器分块） | 3.95 | LDS 每次 FMA 从 2 B 降到 1.5 B |
| 每线程 8 行 | 4.53 | 再降 |
| BM=64 + R=4 | 5.44 | 更大的行块 |
| LDS 读取 4→8 字节（half4） | 5.85 | 访存指令减半 |
| 256 线程 + BM=128 | 6.75 | 8 warps/CU，延迟隐藏 |
| `v_dot2c_f32_f16` 内建 | 6.87 | 两条 fp16 乘加一条指令 |
| **转置 sP（每 key 一次 8 B 读）** | **8.55** | PV 相 LDS 指令再降 4 倍 |

## 失败的尝试

- **寄存器双缓冲预取**（overlap 全局加载）：31.69 ms，比不预取还慢。
  原因：`BN*D/NTHREADS = 16` 个 half × 2 = 32 个额外寄存器，直接导致 spill。
  RDNA2 没有 `cp.async`，靠寄存器暂存反而得不偿失。
- **BN=64**：LDS 需求涨到 44–80 KB，占用率掉到 1 工作组/CU，全面变慢。
- **512 线程**：25.97 ms，比 256 线程差（寄存器压力）。

## 与现状的对比

在 sd.cpp 里实测的注意力约 **2–3 TFLOPS**（22 帧每 token 2.3 GFLOP，占步时接近一半）。
本次微基准达到 **8.5–8.8 TFLOPS**，即 **约 3 倍**提升空间。

按此推算（N=5670，attention 65 TFLOP/步）：
- 现在 attention ≈ 65/2.5 ≈ 26 s/步
- 用 8.76 TFLOPS ≈ 7.4 s/步
- 56 帧步时从 ~50 s 降到 ~31 s

这是「工作流二」接入 sd.cpp 的收益预期，也是通往 30 TFLOPS 端到端的第一步。
