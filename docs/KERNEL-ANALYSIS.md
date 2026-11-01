# 为什么这套 H3 在本机比 CUDA 慢：算子 / kernel / 生态三层归因

> 方法：读 ggml 的 CUDA/HIP 共享源码（`ggml-org/ggml@master`）里的架构分支，
> 结合本机 ROCm 7.2 与 Vulkan 后端的实测数据，逐层定位。
> 所有代码引用都给出行号，实测数据都标注了测量时间窗。

---

## 0. 一句话结论

慢的根因按贡献排序：

1. **硬件**：RDNA2（gfx1030）**没有矩阵核心**（无 MFMA、无 WMMA），INT8 吞吐 46 TOPS，
   而 3090 是 284 TOPS（**6.2×**）；FP16 张量核 71 TFLOPS 且能跑到 ~80% 利用率。
2. **内核分支**：ggml 把 gfx1030 归到 `rdna2` 配置——MMQ 关掉 stream-k、J 上限 64（Ampere 是 128+stream-k）；
   FA 没有 MMA 版本，直接落到**纯 FMA 的 tile 内核**。
3. **生态**：`hipBLASLt` 的 kernel 库里**完全没有 gfx1030**（只有 gfx1100+）；
   rocBLAS 的 gfx1030 库 88 个文件里 54 个是 `fallback`，只有 34 个是调优过的。
4. **数据通路**：`CP_ASYNC_AVAILABLE` 只对 CUDA Ampere+ 定义，HIP 侧**没有异步拷贝流水线**；
   编译产物里也没有 CUDA/HIP Graph（`USE_GRAPHS` 不存在）→ 每步几千次 kernel launch 无法被图捕获。

实测佐证：本机 q4_K 线性层在 Vulkan 上能跑到 **24.2 TFLOPS（46 TFLOPS 峰值的 53%）**，
说明 GEMM 本身不差；真正吃掉时间的是 attention（f32 路径 **0.8 TFLOPS**，占矩阵乘时间 71%）。

---

## 1. 硬件层：RDNA2 没有矩阵核心

| | RX 6900 XT (gfx1030) | RTX 3090 | 比值 |
| --- | --- | --- | --- |
| 矩阵核心 | **无**（VALU FMA） | 张量核（m16n8k16 HMMA） | — |
| FP16 | 46.08 TFLOPS（SIMD，packed） | 71 TFLOPS（张量核 dense） | 1.5× 裸峰值 |
| INT8 | 46.08 TOPS | 284 TOPS | **6.2×** |
| 显存带宽 | 512 GB/s | 936 GB/s | 1.8× |

关键不在裸峰值，而在**利用率**：SIMT FMA 在量化 GEMM 上典型只能跑到峰值 30–50%，
张量核能到 70–80%。所以 Q4_K 这类 int8 点积负载，实际差距会被放大到 **8–12×**。

本机实测（Vulkan perf logger，22 帧 / 2970 token / 2 步）：

| 内核 | 吞吐 | 占 FP16 峰值 |
| --- | --- | --- |
| `q4_K` 线性层 | **24.2 TFLOPS** | **53%** ✅ |
| `f16` 线性层 | 15.8 TFLOPS | 34% |
| `f32` attention 矩阵乘 | **0.8 TFLOPS** | **1.7%** ❌ |

**GEMM 已经接近这类硬件的合理水平；attention 才是黑洞。**

## 2. 代码层：三个"只有 CUDA / 只有新 AMD 有"的宏

### 2.1 矩阵指令：gfx1030 两头不沾

`src/ggml-cuda/common.cuh:276-283`

```c
#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE          // 仅 MI100/MI200/MI300
#endif

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE          // 仅 RX 7000/9000
#endif
```

gfx1030 是 RDNA2：**两个宏都不成立**。对比之下 CUDA 侧有
`VOLTA_MMA_AVAILABLE / TURING_MMA_AVAILABLE / AMPERE_MMA_AVAILABLE`（同文件 285-301 行）。

### 2.2 异步拷贝：HIP 完全没有

`common.cuh:301-303`

```c
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif
```

CUDA Ampere+ 用 `cp.async` 把权重从显存直接搬进 LDS，与计算重叠；
HIP 侧连 RDNA3 都没有这条路径 → 只能"加载—同步—计算"串行，量化 GEMM 的 K 循环无法双缓冲。

### 2.3 MMQ 配置表：gfx1030 走 `rdna2` 分支，且关掉 stream-k

`src/ggml-cuda/mmq.cuh:264-292` 的 `ggml_cuda_mmq_get_config`：

```c
#ifdef GGML_USE_HIP
#elif defined(CDNA)  ... #elif defined(RDNA3) ...
#else
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);   // gfx1030 落这里
#endif
#else
    ... ggml_cuda_mmq_get_config_ampere(type, J, fallback);     // CUDA Volta+ 落这里
```

同一份 Q4_K 配置，RDNA2 与 Ampere 的差别（`mmq-config-rdna2.cuh` vs `mmq-config-ampere.cuh`）：

| 参数 | RDNA2 | Ampere | 影响 |
| --- | --- | --- | --- |
| `stream_k` | **false** | **true** | Ampere 可把 K 维切给 SM 做 split-K，瘦长 GEMM 不浪费 wave |
| 最大 J（n 方向 tile） | **64** | **128** | RDNA2 tile 更小 → 同样的 GEMM 触发更多 block、重用更少 |
| occupancy | 2 | 1 | 寄存器/占用率取舍不同 |

我们 H3 的 GEMM 形状是 `m=28672, n=2430, k=5376` 这类"宽而浅"的矩阵，
恰好最吃 stream-k 和大队列 tile —— 而这两项在 gfx1030 上都是关的。

### 2.4 Flash Attention：落到最老的 tile 内核

`src/ggml-cuda/fattn.cu:681-716`

```c
if (amd_mfma_available(cc) && ...)  return BEST_FATTN_KERNEL_MMA_F16;   // CDNA
if (amd_wmma_available(cc) && ...)  return BEST_FATTN_KERNEL_MMA_F16;   // RDNA3/4
...
// If there are no tensor cores available, use the generic tile kernel:
return BEST_FATTN_KERNEL_TILE;
```

gfx1030 两个条件都不满足 → 走 `fattn-tile`（`fattn-tile.cuh`，纯 FMA，无 MMA）。
CUDA 侧同样的调用会走 `fattn-mma-f16`（Turing+），也就是 tensor core 版本。

这解释了实测里 attention 只有 0.8 TFLOPS（峰值的 1.7%）：**不是算法不对，是内核代际差了两代**。

### 2.5 dp4a 反而是有的

`common.cuh` 的 `ggml_cuda_dp4a`：

```c
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);   // gfx1030 走这里 ✅
```

所以量化点积用的是 `v_dot4_i32_i8`（AMD 的 DP4A 等价物），这部分没有掉队。
瓶颈不在点积指令本身，而在 tile 调度与缺少异步流水。

### 2.6 没有 Graph 捕获

ROCm 后端二进制里搜不到 `USE_GRAPHS` 特征串（CUDA 构建会 `features.push_back({"USE_GRAPHS","1"})`）。
一步 H3 有 **2000+ 个 kernel launch**（实测 2 步 4054 次矩阵乘调用），
没有图捕获就只能逐个 `hipLaunchKernel`，每次 5–10 µs 的提交开销累加成秒级。

## 3. 生态层：AMD 自己的库就没给 gfx1030 做优化

### 3.1 hipBLASLt：零个 gfx1030 kernel

`bin\sd-rocm\hipblaslt\library` 里的架构清单：

```
gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201
```

**完全没有 gfx1030**。hipBLASLt 是 AMD 面向 LLM/Transformer 的新一代 BLAS，
RDNA2 用户直接被排除在外，只能回落到老的 rocBLAS Tensile。

### 3.2 rocBLAS Tensile：gfx1030 有一半是 fallback

`bin\sd-rocm\rocblas\library` 中 `*gfx1030*` 共 88 个文件：

- 34 个是调优过的（`Type_HH` FP16、`Type_SS` FP32 等，带 `.co` + `.dat`）
- **54 个是 `..._fallback_gfx1030.hsaco`** —— 意味着 BF16（BB）、FP8（BS/4xi8I）、
  HPA 变体等类型在 gfx1030 上只有"能跑但没优化"的兜底 kernel

对比：CUDA 的 cuBLAS 对每个架构都给全套 SGEMM/HGEMM/IMMA kernel，没有 fallback 概念。

### 3.3 后果

- 想走"FP16 权重 + BLAS"路线提速：**不行**，hipBLASLt 不支持；只能 rocBLAS Tensile（RDNA2 上是 VALU，无 WMMA）
- 想走 BF16/FP8：gfx1030 上只有 fallback kernel，比 FP16 更慢
- 唯一"有调优"的是 FP16/FP32/INT8 的 Tensile 路径

## 4. 实测层：LoRA 会切换内核路径（本节已于第二轮修正）

> **修正说明**：上一版把进度条上的 `s/it` 当成了累计平均，导致「45 s/步 地板」的错误结论。
> 实际上它是**每步耗时**，首步包含一次性的权重 staging，之后才是稳态。

### 4.1 稳态单步耗时（ROCm，每步实测，非平均）

| 配置 | token | 首步 | 稳态单步 | 备注 |
| --- | --- | --- | --- | --- |
| 864×480 / 5 帧 | 810 | 26.1 s | **5.25 s** | |
| 512×288 / 22 帧 | 864 | 20.1 s | **6.5 s** | |
| 512×288 / 56 帧 | 2016 | 24.2 s | **6.0–7.2 s** | 无 LoRA |
| 864×480 / 22 帧 | 2430 | 33.6 s | **6.3–6.7 s** | 无 LoRA |
| 864×480 / 39 帧 | 4050 | 101.5 s | **26.2 s** | 无 LoRA，超线性起点 |
| 864×480 / 56 帧 | 5670 | 155.6 s | **55.6 s** | 无 LoRA |

2440→4050 token（1.67×）耗时涨 **4.2×**，4050→5670（1.4×）涨 **2.1×** ——
即 n>2.4k 后 `t ∝ n^2.5` 左右，**attention 从 2.4k token 起接管**。

### 4.2 最重要的算子级发现：LoRA 会改变内核路径，且两个后端方向相反

| 场景 | 单步（无 LoRA） | 单步（Turbo LoRA） | 结果 |
| --- | --- | --- | --- |
| ROCm 864×480 / 22 帧 | **6.3–6.7 s** | 47–57 s | **LoRA 慢 7–8×** |
| ROCm 512×288 / 56 帧 | 6.0 s | 19.9 s | LoRA 慢 3.3× |
| Vulkan 864×480 / 22 帧 | 70 s | **11.6 s**（LoRA@0） | **LoRA 快 6×** |

两端结论完全相反：
- **ROCm**：不加 LoRA 走原生 MMQ/dp4a 路径（快）；`at_runtime` LoRA 会把它切到"反量化 + 逐层低秩加"的慢路径。
- **Vulkan**：不加 LoRA 走极慢的 Q4_K shader；加 LoRA 反而切到 F16 GEMM（快）。

因此 `gen.ps1` 现在按后端自动决定是否挂 LoRA：**ROCm 不挂，Vulkan 挂权重 0 的空 LoRA**。

`--lora-apply-mode immediately`（加载时合并）在 Q4_K 上直接失败（sampling failed），不可用。

### 4.3 端到端阶段拆解（864×480 / 22 帧 / 8 步 / ROCm）

| 阶段 | 耗时 | 占比 |
| --- | --- | --- |
| 文本编码（TE，含 17.4 GB staging） | 18.7 s | 20% |
| 采样（8 步 × 6.7 s） | 61.2 s | 67% |
| VAE 解码 | 11.7 s | 13% |
| 权重加载（DiT 11 GB） | 9.9 s | 11% |
| **合计** | **92.4 s** | |

### 4.4 `--mmap` 有效：92.4 s → 84.7 s（8%）

mmap 让 GPU 直接映射权重文件，跳过 host→device 的整块拷贝。
`gen.ps1` 现已默认开启（`-NoMmap` 可关）。

### 4.5 修正后的最优配置矩阵（ROCm，mmap）

| 配置 | 步数 | LoRA | 端到端 |
| --- | --- | --- | --- |
| 864×480 / 22 帧（0.9 s） | 8 | 无 | **84.7 s** |
| 864×480 / 22 帧 | 20 | 无 | **152.3 s** |
| 512×288 / 56 帧（2.3 s） | 8 | 无 | ~90 s |
| 512×288 / 56 帧 | 20 | 无 | ~150 s |
| 864×480 / 56 帧（2.3 s） | 4 | Turbo | 491.8 s |

**相比 2026-10-30 的初始基线 3707 s（56 帧/20 步），现在 480p 2.3 秒片 492 s、
512×288 2.3 秒片 ~90 s，即 7.5–41×。**

### 4.6 步数拐点（ROCm，864x480/22f，无 LoRA+mmap）

| 步数 | 端到端 | 画质 |
| --- | --- | --- |
| 4 | 56.0 s | ❌ 明显崩坏（无蒸馏时 4 步不够） |
| 6 | 68.3 s | ✅ 可接受，与 8 步几乎无差别 |
| 8 | 84.7 s | ✅ 好 |
| 20 | 152.3 s | ✅ 最好（原画质） |

结论：**6 步是质量拐点**，4 步不可用。

### 4.7 VAE 相关开关

--vae-tiling：84.7 s → 81.8 s（约 3.5%，在噪声边缘，未纳入默认）。

### 4.8 已穷尽且无收益的项

| 项 | 结果 |
| --- | --- |
| GGML_CUDA_FORCE_MMQ / GGML_CUDA_MMQ_PREC | **是编译期宏，不是运行时环境变量**，无法使用 |
| 运行时环境变量 | 全库只有 5 个 getenv（ALLREDUCE/DEVICES/CUBLAS_COMPUTE_TYPE/GRAPH_OPT/GRAPH_CHUNK_SIZE），与本负载无关 |
| --lora-apply-mode immediately | Q4_K 上直接 sampling failed，不可用 |
| --max-vram 去掉 | 更慢（56.0 s/步） |
| --params-backend diffusion=ROCm0 钉显存 | 无改善（47.3 s/步） |
| --vae-tiling | ~3.5%，噪声级 |
| rocprof 内核级 profile | 安装器需 1.67 GB，AMD 直连仅 136-342 KB/s，且本轮网络完全中断，未完成 |

### 4.9 仍待验证（下一轮）

1. **TE 小量化**：TE 占 18.7 s（20%），其成本几乎全是 17.4 GB 权重 staging；
   换成 Q2_K_M（12.5 GB）预计省 5-7 s。**需要网络恢复后下载**。
2. VAE 解码 11.7 s：--vae-tile-size 细调（当前只试了默认 tiling）。
3. rocprof 内核级数据（若网络恢复）。

### 4.10 第三轮：FA 在 ROCm 上是负优化（已改默认）

实测（ROCm，无 LoRA，mmap，8 步）：

| 配置 | FA 开 | FA 关 | 差异 |
| --- | --- | --- | --- |
| 864×480 / 22f / 8 步 | 84.7 s | **79.7 s** | -6% |
| 864×480 / 22f / 20 步 | 152.3 s | 152.4 s | 0 |
| 864×480 / 39f / 8 步 | ~26.2 s/步 | **24.1 s/步** | -8% |
| 864×480 / 56f / 8 步 | 55.6 s/步 | **42.2 s/步** | **-24%** |
| 512×288 / 56f / 8 步 | 97.2 s | **83.5 s** | -14% |

关键：**FA 关掉后 5670 token 也没有 OOM**（f32 注意力矩阵约 7.2 GB + 权重 11 GB），
因为 ROCm 报真实显存 + mmap 让权重可映射。此前「长片必须开 FA」的结论只对 Vulkan 成立。

`gen.ps1` 现在：ROCm 默认 FA 关，Vulkan 默认 FA 开。

### 4.11 分辨率-时长成本矩阵（56 帧 = 2.3 s，8 步，FA 关）

| 分辨率 | token | 端到端 | 相对 512 |
| --- | --- | --- | --- |
| 512×288 | 2016 | **83.5 s** | 1.0× |
| 640×384 | 3360 | 233.0 s | 2.8× |
| 864×480 | 5670 | 376.9 s | 4.5× |

实测倍率 2.79×（512→640）与 n² 预测 2.78× 完全吻合 —— **attention 的 n² 是唯一的成本标尺**。

### 4.12 VAE 解码与 TE 的固定成本

- `--vae-tiling`：56 帧 376.9 → **361.0 s**（-4%），逐帧检查无接缝，已设为默认。
- **TE 固定约 18-20 s，且与提示词长度无关**（11 token 与 61 token 都是 ~18 s）
  → 完全是 17.4 GB 权重的 staging，不是计算。
  只能靠更小的 TE 量化（Q2_K_M 12.5 GB，预计省 5-7 s）——**需要下载，当前离线**。

### 4.13 本轮结束时的最优矩阵（ROCm / 无 LoRA / FA 关 / mmap + vae-tiling）

| 配置 | 端到端 | 对比 2026-10-30 基线 |
| --- | --- | --- |
| 864×480 / 22f / 8 步（0.9 s 视频） | **79.7 s** | 892.7 → 79.7（11.2×） |
| 864×480 / 22f / 20 步 | 152.4 s | 892.7 → 152.4（5.9×） |
| 512×288 / 56f / 8 步（2.3 s 视频） | **83.5 s** | — |
| 640×384 / 56f / 8 步 | 233.0 s | — |
| 864×480 / 56f / 8 步（2.3 s 视频） | **361.0 s** | 3707 → 361（10.3×） |

### 4.14 第四轮：运行时环境变量全部试完（全部为 no-op）

这次把 ggml-cuda 里**所有** `ggml_cuda_env_flag/int/set` 读的环境变量都挖出来了
（之前只扫了裸 `getenv`，漏了 helper 读取的），逐个实测（22f/864/8 步/FA 关/mmap，基线 79.7 s）：

| 环境变量 | 语义 | 实测 | 结论 |
| --- | --- | --- | --- |
| `GGML_OP_OFFLOAD_MIN_BATCH=0` | batch 小于阈值的小算子下放 CPU（默认 32） | 79.8 s | 无变化 |
| `GGML_CUDA_REGISTER_HOST=1` | 注册 host 内存加速传输 | 79.8 s | 无变化 |
| `GGML_CUDA_NO_PINNED=1` | 禁用 pinned 内存 | 80.3 s | 无变化 |
| `GGML_CUDA_DISABLE_FUSION=1` | 关闭融合 | — | 只用于 MoE shared-expert 融合，H3 是稠密模型，不适用 |
| `GGML_CUDA_GRAPH_OPT=1` | 图优化/融合 | — | 代码里 `if (!use_cuda_graph) return;`，HIP 上 `use_cuda_graph=false`，永远不生效 |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | rocBLAS 计算精度 | — | F16 已默认走 `CUBLAS_COMPUTE_16F` |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | 统一内存 | 未试 | 会引入 paging，风险大于收益 |

### 4.15 VAE tiling 要按帧数条件启用

| 帧数 | 关 tiling | 开 tiling | 结论 |
| --- | --- | --- | --- |
| 22 | **79.7 s** | 81.6 s | 开反而慢 2% |
| 56 | 376.9 s | **361.0 s** | 开省 4% |

`--vae-tile-size 512x512` 无额外收益（80.3 s）。`gen.ps1` 已改成 **仅 `Frames >= 39` 时加 tiling**。

另外确认：日志里那 3 次 `vae decode graph` 不是预览解码——音频 VAE 只占 0.67 s，
8.0 s（22f）/20 s（56f）是视频 VAE 的真实解码成本，`--preview-interval` 未启用，没有浪费。

### 4.16 采样循环已经贴着内核地板

22 帧单步 6.6 s 的构成估算：

- FFN：2 × 20B × 2430 token = **97.2 TFLOP**；
  按实测 `q4_K` 内核 15–24 TFLOPS → **4.0–6.5 s**
- attention：5.6 TFLOP（f32，非 FA）→ 1–2 s
- 合计理论地板 **5–8.5 s/步**，实测 **6.6 s/步**

**结论：采样循环已经落在估算地板区间内，没有数量级空间了。**
再快只能改内核（重编译 ggml / 等 AMD 补 hipBLASLt 的 gfx1030 支持），不是调参能解决的。

### 4.17 仍未验证的项（全部卡在外部条件）

| 项 | 阻塞原因 |
| --- | --- |
| TE 换 Q2_K_M（预计 -7%） | 需要下载 12.5 GB；网络只剩 download.amd.com 可达 |
| rocprof 内核级 profile | 需要 1.67 GB 安装器，AMD 直连 136–342 KB/s |
| `--tensor-type-rules` 重量化 | 语法探针未生效（q4_0 未出现在权重统计里），且 Q4_K→Q4_0 预期只有 ±10%，风险是加载期重算 |
| 自编译 ggml（开 HIP 图捕获/改 MMQ 配置） | 无编译器工具链（无 MSVC/clang/cmake），HIP 编译器也未随运行时提取 |

### 4.18 `--tensor-type-rules` 对 H3 的 Q4_K 检查点无效

验证方式：`--tensor-type-rules "^blocks\.0\.mlp\.fc1\.weight=f16"` 跑一次 5 帧/1 步
（先修正了一个自身错误：必须同时给 `--cfg-scale 1.0`，否则 cfg=7 会去编码负面提示词而报错）。

结果：进程正常结束（exit 0），但权重统计仍然是 `f16: 102 | q4_K: 208`——**没有任何 tensor 被转换**。
源码里 `parse_tensor_type_rules` / `set_wtype_override` 存在，但对这个已经量化好的 GGUF 检查点
没有生效（推测只影响加载期本就该转换的路径，不会对 Q4_K 现成块做重算）。
因此「把部分层换成 f16/Q8_0 去命中调优 kernel」这条路在现成二进制上不可用。

### 4.19 四个预设的最终验证（2026-11-01 00:50）

| 预设 | 参数 | 实测 |
| --- | --- | --- |
| fast | 864×480 / 22f / 8 步 | 83.8 s |
| draft | 512×288 / 56f / 8 步 | 86.1 s |
| quality | 864×480 / 22f / 20 步 | 155.2 s |
| longhd | 864×480 / 56f / 8 步 | 363.8 s |

### 4.20 网络恢复后：TE 换 Q2_K_M，端到端 -15.8%（已设为默认）

阻塞解除后立刻验证了 TE 量化这一项（原 Q4_K_M 17.0 GB → Q2_K_M 12.5 GB）：

| 指标 | Q4_K_M TE | Q2_K_M TE | 变化 |
| --- | --- | --- | --- |
| 引导图（TE）耗时 | 18.0 s | **6.4 s** | **-64%** |
| 端到端（22f/8 步/FA 关/mmap） | 79.7 s | **68.1 s** | **-14.6%** |
| gen.ps1 fast 预设实测 | 84.8 s | **71.4 s** | **-15.8%** |

比预估的 -7% 好一倍——因为 TE 的成本几乎全是权重 staging，17 GB→12.5 GB 直接砍掉 64%。
画质 A/B（同提示词同 seed）：两个版本都正常，Q2_K 版本对比度略高，无明显提示词遵循退化。
`gen.ps1` 现在默认用 Q2_K_M（不存在时自动回退 Q4_K_M），可用 `-Te` 指定任意 TE。

### 4.21 更新后的最优矩阵（2026-11-01 02:00，Q2_K_M TE + 无 LoRA + FA 关 + mmap）

| 预设 | 参数 | Q4_K TE | Q2_K TE | 变化 |
| --- | --- | --- | --- | --- |
| fast | 864×480 / 22f / 8 步 | 83.8 s | **71.4 s** | -14.8% |
| draft | 512×288 / 56f / 8 步 | 86.1 s | **73.8 s** | -14.3% |
| quality | 864×480 / 22f / 20 步 | 155.2 s | **142.5 s** | -8.2% |
| longhd | 864×480 / 56f / 8 步 | 363.8 s | ~350 s（待测） | ~-4% |

### 4.22 自编译 ggml 的工具链已确认可用（下一步）

网络恢复后发现这台机器其实具备完整编译环境：

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| MSVC 2022 (14.44) | `C:\Program Files\Microsoft Visual Studio\2022\Community` | ✅ |
| Windows SDK 10.0.26100 | `C:\Program Files (x86)\Windows Kits\10` | ✅ |
| CMake 3.31.6 + Ninja | VS 自带 `CommonExtensions\Microsoft\CMake` | ✅ |
| HIP 编译器 clang/hipcc | 从 AMD 安装器提取的 `ROCm_SDK_Core.msi` | ✅ |
| hipblas.lib / rocblas.lib 等导入库 | `ROCm_Libs_Dev.msi` | ✅ |

已用 `-DSD_HIPBLAS=ON -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030` 走到 configure 阶段，
只差 sd.cpp 的 ggml 子模块（`leejet/ggml`，GitHub 归档不含子模块，正在下载）。
可尝试的内核级改动：`GGML_CUDA_USE_GRAPHS=ON`（hipGraph 捕获，省每步数千次 launch）、
修改 `mmq-config-rdna2.cuh` 的 tile/stream-k 参数。

### 4.23 自编译成功，并做了两组内核级实验

工具链凑齐后（MSVC 2022 + Windows SDK + VS 自带 CMake/Ninja + AMD clang/hipcc），
用 sd.cpp `37bc88b` + 钉死的 ggml 子模块 `7b2b435d` 编译成功（3.1 分钟，gfx1030）。

> 关键坑：GitHub 归档不含子模块；ggml 必须用 sd.cpp 钉的那个 commit
> （`master` 缺 `ggml_quantize_i8_convrot` 等补丁函数，编译会直接失败）。
> 另外链接期需要 `ROCm_Libs_Dev.msi` 里的 `hipblas.lib/rocblas.lib` 和
> `hipblas/hipblas.h` 头文件，`ROCm_Libs_RT.msi` 只有 DLL。

自编译产物与官方 release 持平（70.6 s vs 68.1 s，噪声内），说明官方构建没有藏私。

| 内核级实验 | 结果 | 结论 |
| --- | --- | --- |
| 给 Q4_K 打开 **stream-K**（AMD 在 RDNA2 上关着） | 87.1 s（+23%） | ❌ AMD 关得有道理，回滚 |
| 给 Q4_K 加 **J=128 tile**（Ampere 有、RDNA2 缺） | 69.0 / 67.9 s vs 官方 70.6 / 69.7 s | ✅ 稳定 **-2.5%**，保留 |
| 检查 HIP 图捕获 | `GGML_HIP_GRAPHS=ON` 已是默认 | 本来就开着，无遗漏 |

2.5% 不值得替换默认二进制，因此 `gen.ps1` 仍用官方 release；
自编译版本保留在 `build-hip\bin\sd-cli.exe`（需 ROCm SDK 的 bin 在 PATH 里）。

### 4.24 第五轮：内核参数扫描（结论是「官方配置已经很对」）

在自编译基线上逐个改参数，每次「增量编译 ~0.7 min + 基准 1.5–6 min」：

| 改动 | 22f / 8 步 | 56f / 8 步 | 结论 |
| --- | --- | --- | --- |
| 官方 release | 69.7 s | **361.0 s** | 基线 |
| J=128 tile | 67.9 s（-2.6%） | — | 小赢 |
| J=128 + `FORCE_MMQ=ON` | 67.9 s（持平） | **524.4 s（+45%）** | FORCE_MMQ 长序列重创 ❌ |
| J=128 + occupancy 2→1 | **67.3 s（-3.4%）** | — | 小赢 |
| J=128 + occ=1，关 FORCE_MMQ | — | 447.4 s（+24%） | 仍然输给官方 ❌ |
| stream-K 开启 | 87.1 s（+23%） | — | ❌ |

**关键结论：AMD 在 `mmq-config-rdna2.cuh` 里的默认值对长序列（n=5670）是最优的**，
我改的大 J tile / 低 occupancy 只在 n=2430 的小 batch 上快 3%，n 一大就翻车。
`FORCE_MMQ` 更不能开——它会强制 f16 矩阵乘也走 MMQ，56 帧直接慢 45%。

因此 **官方 release 保持为默认**；自编译版本只作为实验产物保留在 `build-hip\bin\`，
且当前配置（J=128 + occ=1）只适合 22 帧的小 batch 场景。

**未做但价值最高的一项**：sd.cpp 的手动注意力路径是 **F32**（`pad_head` 分支显式要求 q/k/v 为 F32）。
把 q/k/v 转 F16 再做 QK^T/AV，能让 attention 的显存带宽和算力各减半——56 帧这种 attention 主导的
配置理论上有 20-30% 空间。这需要改 sd.cpp 源码并重新验证画质，是下一轮的首要目标。

### 4.25 第六轮：f16 注意力实测，以及「噪声地板」的发现

改了 sd.cpp 的手动注意力路径（`src/core/ggml_extend.cpp`），用 `SD_ATTN_F16` 环境变量门控：
把 q/k/v 转 F16 后再做 `QK^T`，softmax 仍 F32，`kq` 再转 F16 做 `AV`。

| 配置 | SD_ATTN_F16=0 | SD_ATTN_F16=1 | 差异 |
| --- | --- | --- | --- |
| 56 帧 / 8 步 | 402.2 s | 392.2 s | -2.5% |
| 22 帧 / 8 步（stock kernel config） | 66.2 s | 66.1 s | 0% |

只快 2.5%，远低于预期。原因：`ggml_prec_set_acc(kq, GGML_PREC_F32)` 仍强制 F32 累加，
外加两次 cast 的开销，f16 只省到一部分带宽。

**更重要的发现是噪声地板**：同一配置（官方 release / 自编译 stock 配置）重复测量得到
69.7 s、68.1 s、66.2 s、70.6 s——**同配置波动约 ±5%**。
这意味着前面测到的 J=128（-2.6%）、occupancy（-1%）全部落在噪声内，不能算真实收益。
只有幅度大的结论是可信的：stream-K **+23%**、`FORCE_MMQ` 在 56 帧 **+45%**、TE Q2_K **-15%**。

### 4.26 算子层面的搜索已到边界（结论）

| 层面 | 已做 | 结果 |
| --- | --- | --- |
| 运行时环境变量 | 全部 8 个（含 helper 读取的） | 全部 no-op |
| CLI 开关 | LoRA 模式/FA/mmap/VAE tiling/tile-size/max-vram/params-backend/sage-attn/offload/tensor-type-rules | 只有 mmap、FA、LoRA 策略有效 |
| 量化 | TE Q4_K→Q2_K、DiT Q3_K、bf16→f16 | 只有 TE Q2_K 有效（-15%） |
| 内核参数 | J tile、occupancy、stream-K、FORCE_MMQ、图捕获 | 只有负面影响（>噪声） |
| 注意力精度 | q/k/v → F16 | +2.5%（噪声边缘） |
| 后端 | Vulkan vs ROCm | ROCm 胜 |
| 编译 | 自建工具链 + 重编译 | 与官方持平 |

采样循环实测 **6.5 s/步（22 帧）**，落在估算内核地板 **5–8.5 s/步** 区间内；
再加上 ±5% 的测量噪声，**继续调参已经没有可观测收益**。

仍不可及的项（需要写新内核或等 AMD 生态）：
- rocprof（这版 HIP SDK 未包含）
- 自定义注意力 kernel / chunked attention（要写 HIP kernel）
- H3 官方稀疏注意力（未开源）
- hipBLASLt 的 gfx1030 支持（AMD 侧缺口）

### 4.27 f32 注意力 0.8 TFLOPS 的根因与两条解法（本轮结论）

**根因（源码级）**：sd.cpp 在手动注意力路径上给 QK^T 显式设了 `GGML_PREC_F32`
（`src/core/ggml_extend.cpp`，`ggml_prec_set_acc(kq, GGML_PREC_F32)`）。而在 ggml 的 Vulkan 后端里：

```cpp
// ggml/src/ggml-vulkan/ggml-vulkan.cpp
static bool ggml_vk_get_mul_mat_mat_f16acc(... ggml_prec prec) {
    return prec == GGML_PREC_DEFAULT && ctx->device->fp16 && ...;
}
```

**只要 prec 不是 DEFAULT，f16 累加的快速 matmul 路径就被彻底关掉**，只剩纯 f32 shader GEMM。
RDNA2 没有 coopmat，f32 GEMM 就是实测的 **0.8 TFLOPS（峰值的 1.7%）**；
再叠加上 n² 的 f32 分数矩阵（2970 token 时约 2 GB）和独立的 scale/softmax 全量扫。

**解法一（零成本，已验证 2.4 倍）**：Vulkan 上直接用 `--flash-attn on` 走 f16 flash 内核。

同一状态下背靠背实测（Vulkan，864×480，22 帧，8 步）：

| 配置 | 采样 | 单步 |
| --- | --- | --- |
| FA 关（走 f32 手动注意力） | 753.1 s | 94.1 s/step |
| **FA 开（f16 flash 内核）** | **310.9 s** | **38.9 s/step** |

**这就是那 0.8 TFLOPS 的直接解法**，不需要编译。`gen.ps1` 在 Vulkan 后端本来就默认开 FA，
所以 Vulkan 用户默认已经避开了这个坑；只有在手动关掉 FA 时才会掉进 f32 路径。

**解法二（源码补丁，已在 ROCm 验证）**：把 q/k/v 转 F16，并且**不再强制 PREC_F32**。

改动（`src/core/ggml_extend.cpp`，用 `SD_ATTN_F16` 环境变量门控）：

```cpp
if (sd_attn_f16_enabled()) { q = cast(q,F16); k = cast(k,F16); v = cast(v,F16); }
auto kq = ggml_mul_mat(ctx, k, q);
if (!sd_attn_f16_enabled()) { ggml_prec_set_acc(kq, GGML_PREC_F32); }  // 关键：不再强制
...
if (sd_attn_f16_enabled()) { kq = cast(kq, GGML_TYPE_F16); }
kqv = ggml_mul_mat(ctx, v, kq);
```

ROCm 上 56 帧（同一二进制、背靠背）：`SD_ATTN_F16=0` 490.6 s → `SD_ATTN_F16=1` **457.4 s（-6.8%）**。
22 帧画质检查无可见退化。

> 说明：ROCm 上收益远小于 Vulkan（6.8% vs 2.4 倍），因为 ROCm 走 rocBLAS，
f32 GEMM 本来就有 Tensile 调优 kernel；Vulkan 没有 BLAS，f32 只能靠 shader，所以特别惨。

### 4.28 严格协议下的复核（2026-11-01 下午）

按工作流四建立协议后重测（同一次会话、背靠背、重复取中位）：

**先说结论：22 帧仍停在 15.4 TFLOPS（33%），没到 30–32。**

模型参数量实测（GGUF 张量统计）：**20.11 B**（mlp 12.02 B + attn 8.02 B + 其余 0.07 B）。
据此 22 帧每步 = FFN 97.7 + attention 5.6 = **103.3 TFLOP**。

| 配置 | 采样 | 单步 | 端到端 |
| --- | --- | --- | --- |
| official（基线，重复 2 次） | 53.2 / 53.3 s | 6.66 s | **15.5 TFLOPS** |
| f16 注意力补丁 | 51.5 / 55.2 s | 6.4 / 6.9 s | 无差异（噪声） |
| `--flash-attn on` | 53.1 s | 6.6 s | 无差异 |
| `--params-backend diffusion=ROCm0`（无 offload） | 50.3 s | 6.3 s | **总耗时 139 s ❌**（TE/VAE 失去 offload） |
| `--offload-to-cpu` + `diffusion=ROCm0` 覆盖 | 50.4 s | 6.3 s | **总耗时 133 s ❌** |

**两个工作流一的项被证伪：**
1. **f16 注意力在 22 帧无效**（53.2 vs 53.35）——22 帧的 attention 只占 5.4% FLOPs，
   且 ROCm 的 f32 路径本来就比 Vulkan 快。
2. **钉显存是净亏**：采样快 6%，但 TE/VAE 失去 offload 后总耗时从 69 s 涨到 133–139 s。
   `--offload-to-cpu` 的整体布局是对的，不存在「每步权重回传」这个待修项。

**成本分解（5 帧 vs 22 帧，钉显存数据）**：810 token 1.9 s/step → 2430 token 6.3 s/step，
即 **2.72 ms/token**，折合 FFN 在真实链路里只有约 **15 TFLOPS**（而不是孤立的 24）。
22 帧时 attention 仅 5.4% FLOPs，但按速率差推算它仍占了近一半时间。

**要跨到 30–32 TFLOPS（≈3.3 s/step）必须同时做到：**
- attention 从 ~2 TFLOPS 提到 15+ TFLOPS（自写 f16 tiled kernel，工作流二）
- FFN 在链路里从 15 提到 22–24 TFLOPS（tile/stream-K/双缓冲，工作流三）

这两项都是**写 kernel**，不是调参数。配置层面已穷尽——这也再次确认了 4.17/4.26 的判断。

### 4.29 56 帧断崖的根因：WDDM 把大张量降级到共享内存（已修复）

用逐算子 HIP event 剖析（只在图末尾同步一次，避免污染）：

| 算子 | 39 帧 | 56 帧 | 每次调用 |
| --- | --- | --- | --- |
| MUL_MAT | 22.9 s / 69% | 40.4 s / 41% | 4.8 → 6.8 ms（正常，×1.42 = token 比）|
| SCALE | 0.51 s / 242 GB/s | **18.8 s / 9.4 GB/s** | 0.21 → 7.8 ms |
| FLASH_ATTN | 6.0 s | 16.3 s / 2.0 GB/s | 8.0 → 15.8 ms |
| CONCAT | 0.49 s / 156 GB/s | 8.9 s / 12.4 GB/s | 0.09 → 1.4 ms |
| GLU | 0.22 s / 130 GB/s | 7.4 s / 5.5 GB/s | 0.73 → 24 ms |

**排除项**（都实测）：host 内存（`host=0` 全为 0）、非连续张量（`nc=0` 全为 0）、
kernel 派发切换（mmvf/mmf/mmvq/mmq/cuBLAS 计数三档完全相同）、系统内存压力（可用 15–17 GB）、
T=16 边界（736×416/56f 同样是 T=17，带宽正常 242 GB/s）。

**根因**：`--offload-to-cpu` 会 staged 约 11.4 GB 权重进 VRAM；当激活张量超过约 600 MB
（= 864×480 / 56 帧的 790 MB）时，超出 Windows 的有效显存预算，**WDDM 把这些分配静默降级到
共享内存**——ggml 视角仍是 VRAM（所以 host 检测为 0），但实际访问走 PCIe，带宽掉到 5–12 GB/s。

**修复**：把 staged 权重显存压到 10 GB，给激活腾出空间：

```
--max-vram ROCm0=10
```

| 配置 | 修复前 | 修复后 | 倍数 |
| --- | --- | --- | --- |
| 864×480 / 56f 采样 | 412.1 s（51.5 s/步） | **147.0 s（18.4 s/步）** | **2.8×** |
| 864×480 / 56f 总耗时 | 444.2 s | **190.2 s** | 2.3× |
| longhd 预设（gen.ps1） | 363.8 s | **191.8 s** | 1.9× |
| 864×480 / 22f | 50.9 s | 63.5 s | ❌ 变慢 25% |

所以**必须按尺寸条件启用**。阈值实测在 583 MB（正常）与 682 MB（塌陷）之间，
对应约 5500 tokens。`gen.ps1` 现在按 `latentT × (W/32) × (H/32) ≥ 5500` 自动加这个参数，
22 帧等小配置不受影响（fast 预设仍 69.9 s）。

### 4.30 MUL_MAT 的剩余空间评估（工作流三结论）

断崖修好之后，MUL_MAT 重新成为最大项：22 帧占 69%，56 帧占 41%。

折算成有效算力：

| 配置 | 每步 GMAC 量 | MUL_MAT 占比 | 折算算力 |
| --- | --- | --- | --- |
| 22 帧 / 8 步 | 97.7 TFLOP | 69% × 6.3 s = 4.35 s | **≈ 22.5 TFLOPS** |
| 56 帧 / 8 步 | 277 TFLOP | ≈85% × 18.1 s = 15.4 s | **≈ 18 TFLOPS** |

而孤立测得的 `q4_K` GEMM 内核是 **24.2 TFLOPS**（Vulkan）/ 相近水平（ROCm）。
也就是说**在链路里已经跑到内核上限的 75–93%**，剩下的差距来自每步的非 GEMM 开销与缓存压力，
不是「GEMM 没调好」。

所以工作流三能做的（都试过了，见 4.24）：

| 手段 | 结果 |
| --- | --- |
| J tile 64→128、occupancy 2→1 | 22 帧 -2.6%（噪声边缘），56 帧 +24%（更差）|
| stream-K | +23% |
| `FORCE_MMQ` | 56 帧 +45% |
| 双缓冲/流水 | RDNA2 无 `cp.async`，寄存器暂存导致 spill（注意力内核上实测更慢）|

**结论：MUL_MAT 上再想拿到两位数收益，需要的是一个比现有 MMQ 更快的 q4_K GEMM 内核
（研究级工作），或者带矩阵核心的硬件。调参已经到头。**

按 4.16 的地板估算：22 帧单步 6.3 s 里，FFN 理论地板（97.7 TFLOP ÷ 24 TFLOPS）≈ 4.1 s，
实测 4.35 s —— **已经在 94% 的地板位置**。

### 4.31 MMQ q4_K 内核结构（写新 GEMM 前的必修课）

读 `ggml/src/ggml-cuda/mmq.cuh` + `mmq-load-tiles.cuh` 得到的事实：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `MMQ_ITER_K` | 256 | 每次 K 迭代处理一个 QK_K 超级块 |
| `MMQ_TILE_NE_K` | 32 | dp4a 路径的 K 粒度 |
| `QK_K` | 256 | q4_K 超级块大小（8 个子块 × 32 值） |
| `K_SCALE_SIZE` | 12 | 8 个子块的 6-bit scale/min 打包 |
| RDNA2 配置 | nthreads=256, occ=2, I=128, J=8…64, stream_k=false | `mmq-config-rdna2.cuh` |

**dp4a 的 SRAM 布局**（`MMQ_DP4A_TXS_Q4_K`）：

```
qs = I*32 + I        // 反量化后的 int8 值
dm = I*32/QI4_K      // scale/min（half2）
sc = I*32/8 + I/8    // 子块 scale
```

I=128 时 qs ≈ 4 224 int = **16.9 KB**，加上激活侧 Q8_1 tile，单个 workgroup 约 30–40 KB
→ 每 CU 只能放 2 个 workgroup（64 KB LDS 上限）。

**反量化在 tile 加载阶段完成**（每块一次），内层全是 `dp4a`：

```cpp
x_qs[...] = (qs0 >> 0) & 0x0F0F0F0F;   // 低 4 位 → int8
x_qs[...] = (qs0 >> 4) & 0x0F0F0F0F;   // 高 4 位 → int8
```

### 4.32 修正：GEMM 调参**没有**到头 —— J=128 是真实收益

4.30 节里我写过「MUL_MAT 已在内核地板上，调参到头」。**这个结论是错的**，原因是我用错了
形状假设（拿 K=7168 去测，而模型真实的是 K=5376），以及早前在 sd.cpp 层做 A/B 时被噪声淹没。

用独立微基准（`hip/gemm_bench.cpp`，驱动 ggml 自身的后端内核）测**模型的真实形状**
（从 GGUF 读出：`mlp.fc1 = [5376, 28672]`、`fc2 = [28672, 5376]`、`qkv = [5376, 21504]`）：

| 形状 | 原厂配置 | + J=128 行 | 提升 |
| --- | --- | --- | --- |
| fc1 22f (5376×28672×2835) | 25.88 TFLOPS | **27.92** | +7.9% |
| fc1 56f (5376×28672×6885) | 26.05 | **28.59** | +9.7% |
| fc2 22f (28672×5376×2835) | 24.24 | **25.59** | +5.6% |
| qkv 22f (5376×21504×2835) | 26.00 | **27.88** | +7.2% |

**根因**：`mmq-config-rdna2.cuh` 里 Q4_K 的 J 只到 **64**（而 J 的编译期上限是 128，
`static_assert(J <= 128)`）。宽 M（28672）的 GEMM 权重会被按 N/J 重复读 45 次，J=128 直接减半。

配置扫描（fc1 形状）：

| 变体 | TFLOPS |
| --- | --- |
| 原厂 | 25.88 |
| **J=128** | **27.92** ✅ |
| J=128 + occupancy 1 | 27.88（无差别）|
| I=64 + J=128 | 22.96（更差）|
| J=256 | 编译失败（`static_assert(256 <= 128)`）|

端到端（同会话背靠背）：

| 配置 | 官方 release | J=128 自编译 | 提升 |
| --- | --- | --- | --- |
| 864×480 / 22f / 8 步 | 50.5 s | **49.2 s** | -2.6% |
| 864×480 / 56f / 8 步 + max-vram 11 | 145.3 s | **138.4 s** | **-4.7%** |

**给上游的线索**：这是一行配置改动（4.32 的表格），对任何 gfx1030 用户都有效，
适合提 PR 给 ggml（`mmq-config-rdna2.cuh` 加两行 J=128）。

### 4.33 修复之后 FA 的结论反转了（+25%）

WDDM 那个坑修好之后重测 56 帧的 flash-attention：

| 配置 | 采样 | 单步 |
| --- | --- | --- |
| 默认（`auto`） | 184.4 s | 23.0 s/step |
| `--flash-attn on` | **138.4 s** | **17.3 s/step** |
| `--flash-attn off`（显式） | **直接失败**（`sampling failed`），手动路径要物化 n² 矩阵，显存不够 |

**修复前的结论是「ROCm 上 FA 关更快」，现在是「FA 开快 25%」**——因为之前正是
WDDM 降级把手动注意力路径打得最惨。22 帧仍然中性（49.5 vs 49.7 s）。

`gen.ps1` 已改为：token 数 ≥ 4000 时强制 `--flash-attn on`（不再依赖 sd-cli 的 `auto`，
它在 56 帧会选到慢路径/失败路径）。

### 4.34 其他量化类型的 J=128 扫描（PR #1677）

在 gfx1030 上，`mmq-config-rdna2.cuh` 里**所有 22 种量化类型的 J 都停在 64**，
而内核允许 J≤128。逐个实测（形状 5376×28672×2835）：

| 类型 | 原厂 | J=128 | 结论 |
| --- | --- | --- | --- |
| Q4_K | 25.88 | **27.92** | **+7.9%** ✅ |
| Q6_K | 21.19 | **24.46** | **+15.4%** ✅ |
| Q4_0 | 36.47 | **40.27** | **+10.4%** ✅ |
| Q5_K | 27.93 | 27.96 | 无变化 |
| Q3_K | 27.89 | 27.94 | 无变化 |
| Q2_K | 5.80 | 6.05 | 噪声内 |
| **Q8_0** | 32.27 | **26.96** | **−16.5%** ❌ 不改 |

**类型相关的**，不能一刀切。Q8_0 已经是纯 int8 路径，加大 J 反而破坏了它的调度。

已提 PR 给上游：https://github.com/ggml-org/ggml/pull/1677
（Q4_K + Q6_K + Q4_0 三组，附带 Q5_K/Q3_K/Q2_K/Q8_0 的负结果，说明为什么不改）

## 5. 与 CUDA 的量化差距（估算）

以 Q4_K GEMM 为例：

| 环节 | 3090 (CUDA) | 6900 XT (ROCm) | 倍率 |
| --- | --- | --- | --- |
| 点积吞吐 | 284 TOPS（tensor core） | 46 TOPS（v_dot4） | **6.2×** |
| 流水线 | `cp.async` 双缓冲 + stream-k | 无异步拷贝 + 无 stream-k | 再 +1.3–1.8× |
| 实际 GEMM 效率 | 峰值 70–80% | 峰值 30–50% | 再 +1.5–2× |

attention（H3 的主要成本）差异更大：CUDA 走 `fattn-mma-f16`，
gfx1030 走 `fattn-tile`（纯 FMA）——**这是代际差，不是调参能补的**。

综合估算：**同等 CUDA 卡（3090 级）在这套 H3 上大约快 6–10×**，
其中硬件占 ~6×，内核代际再叠加 ~1.5–2×。

## 6. 如果想在 gfx1030 上再快，代码层还有什么可动

按"能改代码"程度排序：

1. **绕开 attention 的 tile 内核**：把 attention 拆成块（chunked attention），
   每块序列更短、tile 利用率更高 —— 需要改 sd.cpp 的 H3 图，或改用支持 chunked attention 的运行时（ComfyUI 的 KJNodes 有类似节点）。
2. **压缩 attention 的 f32 通道**：`f32` 分数矩阵让显存和带宽都翻倍；
   若能把 QK^T/AV 固定为 f16（`GGML_CUDA_MMQ_PREC`/graph precision 相关开关），理论上有 2× 空间。
3. **开启 HIP 图捕获**：本机构建没有 `USE_GRAPHS`；
   用 `-DGGML_CUDA_USE_GRAPHS=ON` 自编译可省掉 2000 次/步的 launch 开销（预计 5–10%）。
4. **换 Q8_0 权重**：gfx1030 的 Tensile 有 INT8 调优 kernel，
   而 Q4_K 走 MMQ SIMT；Q8_0 权重更大（21 GB，放不下）——只能在"部分层"上尝试。
5. **ROCm 侧不存在"调个参数就好"的开关**：
   `GGML_CUDA_FORCE_MMQ`、`GGML_CUDA_MMQ_PREC`、`GGML_CUDA_DISABLE_FUSION` 都试过，无稳定收益。

## 7. 结论

- **不是 sd.cpp 写得差**，而是 ggml 的 HIP 后端在 RDNA2 上只能走"无矩阵指令 + 无异步拷贝 + 无 stream-k"的老路径，
  而 CUDA 侧走的是张量核 + 异步流水 + 图捕获。
- **也不是 ROCm 装错了**：设备识别、dp4a、rocBLAS 调优 kernel 都正常工作；
  是 AMD 的 gfx1030 生态（hipBLASLt 零支持、Tensile 半 fallback）本身就没为 LLM 负载投入。
- **可动的空间**在 attention：它是唯一"1.7% 峰值"的内核，也是唯一还有 5–10× 差距的地方；
  改动点在 sd.cpp/ggml 的 H3 attention 图，而不是启动参数。
