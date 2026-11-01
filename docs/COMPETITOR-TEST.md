# 竞品对比：oobabooga 的 sd.cpp 优化分支（gfx1030 实测）

日期：2026-11-02
结论：**在 RX 6900 XT 上，我的配置比这个"更强的"分支快 30%（22 帧）到 8%（56 帧）。**

---

## 背景

全网搜索 MiniMax-H3 的优化方案时，找到 [oobabooga/stable-diffusion.cpp PR #2](https://github.com/oobabooga/stable-diffusion.cpp/pull/2)。
它比上游多做了一整套优化：

| 优化 | 说明 |
|---|---|
| VAE decoder 内 flash attention | H3 视频 VAE 是 36 层 ViT，上游的 `--diffusion-fa` 够不到它 |
| tile 拼接缓存重写 | 内层循环跨 plane，105 个 tile 全部 cache/TLB miss（每次解码 ~10 s host 时间）|
| tile 异步合并 / 分段时间轴异步组装 | 一个 tile 算时另一个线程把上一个融进画布 |
| 音频 VAE 直接深度卷积 | 替代 F16 im2col + matmul |
| DiT 融合算子 | RoPE+重排+KV scale 合一、adaLN 原地、gated residual 原地（逐位相同）|
| 长序列 FA 内核 | "H3 形状最快 4 倍"（Ampere 起）|
| ggml-cuda 补丁 ×6 | BF16 cuBLAS、`cudaGetDeviceProperties` 缓存、向量化行拷贝、cuDNN attention |

## 我怎么测的

1. 拉他的 master（`4a98feb`），ggml 用他钉的 `f583f393`
2. **应用全部 6 个 ggml 补丁**（全部干净应用）
3. 用同一套 MSVC + AMD clang + ROCm 7.2 工具链编 HIP 版（2.2 分钟，一次成功）
4. 同一会话、同一提示词、同一 seed、同一 `--mmap` 背靠背对比

## 结果（8 步，同会话）

| | oobabooga 分支 | 我的配置 | 差距 |
|---|---:|---:|---:|
| 22 帧 采样 | 79.1 s（9.9 s/步） | **52.7 s（6.6 s/步）** | 我快 **33%** |
| 22 帧 总耗时 | 96.7 s | **67.7 s** | 我快 **30%** |
| 56 帧 采样 | 164.7 s | **151.5 s** | 我快 8% |
| 56 帧 总耗时 | 194.0 s | **179.4 s** | 我快 8% |
| 56 帧 VAE 解码 | 22.55 s | **20.5 s** | 我快 9% |

## 为什么会这样

| 原因 | 证据 |
|---|---|
| **他的优化是 N 卡导向的** | cuDNN fused attention（N 卡专用）、长序列 FA（"Ampere 起"）、BF16 cuBLAS（RDNA2 的 BF16 路径慢）——在 gfx1030 上都不适用 |
| **默认 compute buffer 太大，先踩了 WDDM 坑** | 他的版本 22 帧 compute buffer **2478 MB**（我的 1232 MB），不加上限时 **107 s/步**；加 `--max-vram 12` 才回到 15.5 |
| **VAE FA 在 RDNA2 上没赚** | 他的 VAE FA 打开后解码 22.55 s，我的（无 FA）20.5 s——和我在 Vulkan 上测到的"FA 反而慢"一致 |
| **上游更新带来 AMD 回退** | 去掉全部 ggml 补丁后仍然是 107 s/步（WDDM 因素），说明不是补丁的问题 |

## 可以直接抄的两条（还没做）

他的 VAE 侧**host 重叠**思路我确实没做，而且和内核无关：

1. **tile 异步合并**——一个 tile 在算，另一个线程把上一个融进画布（`SD_TILE_ASYNC_MERGE`）
2. **分段时间轴异步组装**——解码下一段时，上一段在 worker 线程做裁剪/交叉淡化（`SD_H3_VAE_ASYNC_ASSEMBLY`）

这两条针对的是"host 时间"，而我这边的 56 帧 VAE 解码 20.5 s 里有多少是 host 时间**没测过**。值得单独测量。

## 社区全景：没有可比基准

| 来源 | 有 gfx1030 的 H3 数据吗 |
|---|---|
| [awesome-minimax-H3 性能指南](https://github.com/wildminder/awesome-minimax-H3)（2 硬基准 + 17 社区报告） | ❌ 全是 N 卡/Apple/DGX |
| [willitrunai](https://willitrunai.com/can-run/hy3-on-rx-6900-xt-16gb) | ❌ 自动估算判定"跑不了"（被本机实测证伪）|
| [A 卡 ComfyUI 教程（B 站转）](https://www.kongchang.com/articles/aqia-win11ben-di-bu-shu-minimax-h3-wen-sheng-tu-tu-sheng-tu-duo-tu-sheng-tu-shi) | ⚠️ "6900 XTX 级别"，但测的是**文生图**（首图 188 s、缓存 103 s），不是视频 |
| [rocmate gfx1030](https://t0nd3.github.io/rocmate/gfx1030/) | ❌ 只有通用建议 |

**所以「6900 XT + H3 视频」这个组合，目前公开资料里没有别的数据点。**
我的 174.6 s（864×480 / 2.3 秒片 / 8 步）是这个组合下唯一的实测数字。
