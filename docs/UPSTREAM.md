# 上游提交记录（ggml / llama.cpp）

这份文档记录向 ggml 生态提交 J=128 优化的完整经过，包括踩到的坑，
免得以后重复犯错。

---

## 1. 提交内容

在 `mmq-config-rdna2.cuh`（RDNA2 的 MMQ tile 配置表）里为 4 种量化各加两行 `J=128`：

```cpp
CASE(GGML_TYPE_Q4_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
CASE(GGML_TYPE_Q6_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
CASE(GGML_TYPE_Q4_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
CASE(GGML_TYPE_Q4_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
```
（每种还要一行 `fallback=false` 的副本）

## 2. 上游实测收益（RX 6900 XT / ROCm 7.2 / Windows）

| 类型 | M=28672 | M=21504 | M=14336 |
|---|---:|---:|---:|
| Q4_K | +7.5% | +7.3% | +7.1% |
| Q6_K | +13.3% | +13.1% | +12.9% |
| Q4_0 | +9.0% | +9.4% | +9.2% |
| Q4_1 | +8.6% | +8.8% | +8.4% |

**排除的类型**（同样实测）：Q8_0 −19.7%、Q2_0 −17.5%、Q5_0 −16.0%、Q3_K/Q5_K 无变化。

## 3. 三个必须知道的坑

### 坑 1：SRAM layout 是**按类型**的，不能照抄

上游 `mmq-config-rdna2.cuh` 里：

| 类型 | 必需 layout |
|---|---|
| Q4_K / Q4_1 / Q5_K | `LAYOUT_Q8_1` |
| **Q6_K** | **`LAYOUT_Q6_K`** |
| **Q4_0 / Q5_0 / Q2_0 / Q1_0** | **`LAYOUT_Q8_0`** |

我第一次提交时从 sd.cpp 自带的老版 ggml 抄了 `Q8_1`（那里所有类型都是 Q8_1），
导致 **Q6_K 和 Q4_0 两行是错的**——如果被合并会喂给 loader 错误的 tile 布局。
**新增行必须和同类型现有行的 layout 逐字一致。**

### 坑 2：J 的选择规则不是"ncols ≥ 128"

`ggml_cuda_mmq_get_J()`（mmq.cu）：

```cpp
for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
    config = get_config(type, J, fallback, cc, prec);
    if (config.type == GGML_TYPE_COUNT) continue;            // 表里没这个 J
    if (mmq_get_nbytes_shared(config, cc) > smpbo) continue; // 显存放不下
    ntiles_x = (ncols_opt + config.J - 1) / config.J;
    if (ntiles_x < ntiles_J_best) { J_best = J; ... }
}
```

按 J 递增找 tile 数最少的配置，**一旦某个 J 让 tile 数变成 1 就停止**。
由于中间宽度（72/80/…/120）表里没有会被跳过，加了 J=128 之后
**只要 `ncols > 64` 就会跳用 128**（tile 数 2→1）。

### 坑 3：MoE 会回退

llama.cpp #27558 的作者报告：J 能到 128 后 **MoE 模型性能回退**。
MoE 的 `ncols_opt` 是"每专家平均 token 数"，同样 > 64 就会选中 J=128。
缓解方案是在 launcher 里按 `ne02 == 1`（dense）门控。

## 4. 提交状态（截至 2026-11-03）

| 项 | 状态 |
|---|---|
| [ggml-org/ggml#1677](https://github.com/ggml-org/ggml/pull/1677) | **打开，0 评论 0 审查**；CI 三个 workflow 卡在 `action_required`（首次贡献者需维护者批准才跑）|
| [llama.cpp#27558](https://github.com/ggml-org/llama.cpp/pull/27558) | 别人的同主题 PR，**已开 73 天**，维护者 IMbackK 有 4 条审查意见（核心争议就是 MoE）|
| [llama.cpp#27555](https://github.com/ggml-org/llama.cpp/pull/27555) | 同作者的重复投递，已关闭 |
| 我在 #27558 的回帖 | [issuecomment-6594263470](https://github.com/ggml-org/llama.cpp/pull/27558#issuecomment-6594263470)，附 4 种类型的上游实测 + 指出 Q5_K 数据不一致 |

## 5. 教训：**ggml-org/ggml 是镜像仓库**

最近合并的 PR 几乎全是 `sync : llama.cpp` 和 `bump version`：

```
#1675 sync : llama.cpp     #1676 bump version to v.28.0
#1659 bump version 0.27.0  #1657 sync : llama.cpp
```

**CUDA/MMQ 内核的工作应该在 llama.cpp 提**，ggml-org/ggml 只是下游同步。

## 6. 测量陷阱（会直接导致错误结论）

1. **动态链接的基准程序会从 `build\bin` 加载 DLL**。重建 DLL 后，"原厂"那份
   实际跑的是打过补丁的版本——表现是两组数字完美重合（27.65 vs 27.62）。
   **必须准备两份隔离的 DLL 目录。**
2. **类型名大小写敏感**。传 `q6_k`（小写 k）匹配不到 `q6_K`，程序**静默回退到
   Q4_K**，于是测出来的是 Q4_K 的数字。任何 `strcmp` 式分发都要确认命中。
3. **本地基线可能过时**。我用的是 sd.cpp 自带的老版 ggml，Q6_K 原厂只有 21.19；
   上游 master 已经是 25.74。直接用本地基线算出的"提升"会虚高。
   **提交前必须用目标仓库的代码重测。**
