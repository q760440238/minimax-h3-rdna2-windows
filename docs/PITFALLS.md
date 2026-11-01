# 踩坑记录（MiniMax-H3 on RX 6900 XT / Windows）

本项目从 0 部署到 19 倍提速过程中踩到的所有坑，按类别整理。每条都注明**现象**和**根因**，
避免下次再花同样时间。

---

## 一、网络与下载

### 1.1 HTTPS 全部返回 000，误判为「代理挂了」

- **现象**：`curl https://modelscope.cn` 返回 `000`，`hf-mirror`/`github` 同样 000，
  但 `Test-NetConnection modelscope.cn:443` 是通的；一度以为是 Clash 节点失效，浪费两轮排查。
- **根因**：Windows 自带 curl 用 schannel，握手时会做证书吊销检查（CRL/OCSP），
  而吊销服务器本身访问不了 → 握手失败。**不是网络问题**。
- **解决**：所有 curl 加 `--ssl-no-revoke`。加完后 ModelScope 直连 25 MB/s。

### 1.2 GitHub 归档 zip 不含 submodule

- **现象**：`stable-diffusion.cpp` 源码编到一半报 `ggml_quantize_i8_convrot` 未声明。
- **根因**：GitHub 的 `archive.zip` 只有主仓库，`ggml/` 是 submodule，解压出来是空目录；
  随便下个 ggml master 又和主仓库版本不匹配（缺私有补丁函数）。
- **解决**：用 GitHub API 查主仓库该 commit 的 tree，取 `ggml` 条目里的**确切 SHA**
  （本项目是 `7b2b435d`），再下那个 SHA 的归档：
  `https://gh-proxy.com/https://github.com/leejet/ggml/archive/<sha>.zip`

### 1.3 镜像站速度差异巨大

- `gh-proxy.com` 稳定 ~120 KB/s，`ghproxy.net` ~50 KB/s，`hub.gitmirror.com` 不通。
- ModelScope 直连最快（25 MB/s），AMD 官网只有 5 MB/s。

---

## 二、ROCm / HIP 在 Windows 上的部署

### 2.1 `0xC0000135 STATUS_DLL_NOT_FOUND`

- **根因**：官方 ROCm Windows 包只带 `sd-cli.exe`，缺 `hipblas.dll` 等运行时。
- **解决**：从 AMD HIP SDK 安装器里免管理员提取 —— 文件是 NSIS，7z 载荷偏移 `1785344`，
  里面 `ROCm_Libs_RT.msi` 用 `msiexec /a` 解包（不需要管理员），把 DLL 复制到 exe 旁边。

### 2.2 编译期找得到 `amdhip64.lib` 却找不到 `hipblas.lib`

- **根因**：`ROCm_SDK_Core.msi` 只有 HIP 运行时和编译器；`hipblas/rocblas` 的
  **导入库 + 头文件 + CMake config** 在另一个包 `ROCm_Libs_Dev.msi` 里。
- **另一个坑**：CMake 的 imported target 还会检查 `bin/hipblas.dll` 是否存在，
  所以必须把运行时 DLL 也复制进 SDK 的 `bin/`，否则 `find_package(hipblas)` 报
  "contained hipblas-targets.cmake but not all the files it references"。
- **解决**：合并三处内容到同一个 SDK 根目录（Core + Libs_Dev 的 lib/include/cmake + RT 的 DLL）。

### 2.3 `hipcc` 报 `HIP_PATH ... 7.2 \bin` 路径带空格

- **根因**：批处理里 `set HIP_PATH=%SDK% && ...` 的 `&&` 前有空格，值里多了尾随空格。
- **解决**：`set "HIP_PATH=%SDK%"`，或者干脆不用 hipcc，直接用 clang：

```
clang.exe -x hip --offload-arch=gfx1030 \
  --rocm-path=<sdk> --hip-device-lib-path=<sdk>\amdgcn\bitcode \
  --driver-mode=g++ -fuse-ld=lld --ld-path=<sdk>\bin\lld-link.exe \
  -I<sdk>\include -L<sdk>\lib --hip-link -O3 ...
```

`--hip-device-lib-path` 指向 `amdgcn\bitcode`（里面有 `ocml.bc`），不是 `lib`。

### 2.4 Windows TDR：kernel 超过约 2 秒被杀

- **现象**：写了个朴素注意力 kernel（N=2430 全串行），运行报
  `HIP error unspecified launch failure`。
- **根因**：Windows 显示驱动超时检测（TDR）默认约 2 秒，超时重置设备。
- **解决**：微基准要么分块，要么只在小尺寸跑基线；生产 kernel 必须控制在秒级以内。

---

## 三、ggml / sd.cpp 的行为陷阱

### 3.1 Vulkan 后端报告「可用显存 0 MB」

- **现象**：`reported free 0.00 MB / total 16368 MB`，容量检查直接拒绝在 GPU 上跑。
- **解决**：显式给预算 `--max-vram Vulkan0=15`。ROCm 后端没有这个问题。

### 3.2 `--params-backend diffusion=ROCm0` 采样更快但总耗时爆炸

- **现象**：采样快 6%（50.3 vs 53.6 s），但总耗时从 69 s 涨到 **133–139 s**。
- **根因**：去掉 offload 后 TE/VAE 失去 staging 安排，模型加载/编码阶段被拖垮。
- **结论**：`--offload-to-cpu` 的整体布局是对的，不要为了采样的 6% 去动它。

### 3.3 `--lora-apply-mode immediately` 在 Q4_K 上直接失败

- 报 `sampling failed`，无法使用。LoRA 只能用默认的 `at_runtime`（ROCm 上还慢 3–7 倍）。

### 3.4 `--sage-attn` 在 ROCm 不可用

- 报错要求 patched GGML + CUDA 12 + SM80，AMD 上没戏。

### 3.5 `--tensor-type-rules` 对已量化的检查点无效

- 现象：给了 `^blocks\.0\.mlp\.fc1\.weight=f16`，进程正常结束，但权重统计没变
  （f16 仍 102、q4_K 仍 208）。
- 结论：这个选项只影响加载期本就要转换的路径，不会对现成 Q4_K 块重算。

### 3.6 进度条的 `s/it` 是**单步**时间，不是累计平均

- 曾据此误判「45 秒/步地板」，实际是首步慢导致的累积平均被读错。

### 3.7 短提示词会慢 2.5 倍

- 实测 <25 词的提示词比长提示词慢约 2.5 倍（250 s vs 124 s）。`gen.ps1` 现在会警告。

### 3.8 cfg 默认 7.0 会去编码负面提示词

- 忘了 `--cfg-scale 1.0` 时，短提示词会报 `failed to encode negative video prompt`
  并中止。H3 是蒸馏模型，必须 cfg=1。

---

## 四、测量陷阱（最贵的几课）

### 4.1 跨时段状态漂移最高 30%

- 同一配置（56 帧 / 8 步）在不同时间测得 361 s、412 s、465 s、524 s。
- **同一次会话内背靠背**重复则很稳（官方二进制 53.2 / 53.3 s，波动 0.2%）。
- **结论**：任何对比必须同会话交替跑 A/B，不能拿昨天的数当基线。

### 4.2 逐节点 `hipEventSynchronize` 会污染小算子

- 现象：SCALE 从 0.21 ms/call 变成 7.6 ms/call，看起来像 40 倍退化。
- **根因**：每个节点强制同步，把 launch/同步开销算进了小算子。
- **解决**：enqueue 事件对，只在**图末尾同步一次**再统一求差。
  （即便如此，结论没变——退化的确存在，但数值要这样测才干净。）

### 4.3 逐节点事件与 HIP 图捕获互斥

- 现象：`ROCm error: operation failed due to a previous error during capture`。
- **解决**：剖析时必须 `-DGGML_HIP_GRAPHS=OFF`；测完记得编回 ON，否则性能数据偏低。

### 4.4 插桩版不能当性能基线

- 曾用插桩版（图捕获关闭 + 剖析开销）测出「自编译版 56 帧慢 24%」，
  清掉插桩重编后实际是**持平**。教训：结论必须来自无插桩的干净构建。

---

## 五、那次 2.8 倍的关键坑：WDDM 静默降级

### 5.1 现象

864×480 / 56 帧每 token 成本从 2.3 ms 跳到 7.1 ms（3 倍），逐算子看：

| 算子 | 39 帧 | 56 帧 |
| --- | --- | --- |
| MUL_MAT | 4.8 ms/call | 6.8 ms/call（正常，随 token 线性）|
| SCALE | 242 GB/s | **9.4 GB/s** |
| CONCAT | 156 GB/s | **12.4 GB/s** |
| GLU | 130 GB/s | **5.5 GB/s** |

### 5.2 排查中排除掉的错误方向

| 假设 | 怎么排除的 |
| --- | --- |
| 注意力 f32 太慢 | FA 开/关 56 帧差 1% |
| kernel 派发切换慢路径 | 插桩统计 mmvf/mmf/mmvq/mmq/cuBLAS 调用数，三档完全相同 |
| 张量在 host 内存 | 给剖析加 `ggml_backend_buft_is_host()`，结果 `host=0` 全为 0 |
| 非连续张量 | 同上加 `ggml_is_contiguous()`，`nc=0` 全为 0 |
| 系统内存压力 | 可用内存 15–17 GB，工作集稳定 22.8 GB |
| T=16 的时间维边界 | **证伪**：736×416/56f 同样是 T=17，带宽 242 GB/s 完全正常 |

### 5.3 真正的根因

约 11.4 GB 权重常驻显存后，超过约 **600 MB** 的激活张量超出 Windows 有效显存预算，
**WDDM 把这些分配静默降级到共享内存**。从 API 视角它还在 VRAM（所以 `host=0`），
但实际访问走 PCIe → 带宽掉到 5–12 GB/s。

### 5.4 修复与边界

`--max-vram ROCm0=11`（压低权重常驻，给激活腾地方）：56 帧 412 s → **145 s**（2.8 倍）。

**这个值是双边有界的**（实测）：

| max-vram | 56 帧采样 |
| --- | --- |
| 9 GB | 147.0 s |
| 10 GB | 145.7 s |
| **11 GB** | **145.3 s** |
| 12 GB | 144.6 s |
| **13 GB** | **400.0 s** ❌ 断崖复现 |

所以规律不是「越大越好」，而是**给激活留够 ~4–5 GB**。
换 24 GB 卡时应按 `显存 − 5 GB` 估算（例如 19 GB），而不是照抄 11。

---

## 六、PowerShell 相关（写脚本时踩的）

| 坑 | 现象 | 解决 |
| --- | --- | --- |
| `$ErrorActionPreference='Stop'` | sd-cli 的 stderr 被当成致命错误 | 改 `'Continue'` |
| `$args` 是保留变量 | 参数数组赋值后行为异常 | 改名 `$cliArgs` |
| `"$Out.log"` | 被解析成 `$Out` 属性 `.log` | 用 `"${Out}.log"` |
| 双引号 here-string 里的 `` `b `` | **插入退格符 U+0008**，把 README 里的路径打坏 | 用单引号 here-string 或 `write` 工具 |
| PowerShell 5.1 读写 UTF-8 | 中文变乱码 | 用 `write` 工具或 Python `encoding='utf-8'` |
| 直接 `& script.ps1` | 执行策略禁止 | `powershell.exe -File ... -ExecutionPolicy Bypass` |
| 数组参数经 `powershell.exe -File` 传递 | 被展平/解析错 | 改传逗号分隔字符串 |

---

## 七、内核调优的无效方向（省得再试）

| 尝试 | 结果 |
| --- | --- |
| Q4_K 打开 stream-K | 87.1 s vs 70.6 s（**+23%**），AMD 关掉是对的 |
| J tile 64→128、occupancy 2→1 | 22 帧 -2.6%，56 帧 +24%，噪声与副作用并存 |
| `GGML_CUDA_FORCE_MMQ=ON` | 56 帧 +45%，强制 f16 也走 MMQ 是灾难 |
| `GGML_CUDA_FORCE_CUBLAS=ON` | 未单独测（MMQ 更快） |
| `--tensor-type-rules` 重量化 | 对现成 Q4_K 无效 |
| 8 个 CUDA/HIP 环境变量 | 全部 no-op（`OP_OFFLOAD_MIN_BATCH`/`REGISTER_HOST`/`NO_PINNED` 等）|
| Vulkan 上 f32 注意力 | 开 FA 快 **2.4 倍**（`PREC_F32` 会关掉 f16 累加路径）|
| ROCm 上开 FA | 反而慢，且 56 帧与关 FA 只差 1% |

---

## 八、补充（第二轮复盘补录）

以下几条第一版漏掉了，全部在本会话中真实发生过。

### 8.1 两个 FLOP 口径错误：token 数与「每层 vs 每步」

- **错误一（token 数）**：一开始用 `tokens = frames/4 × H/32 × W/32`，把 56 帧算成 5670 token。
  日志里真实的 latent 是 `54x30x17x24`，实际 token = 17 × 27 × 15 = **6885**。
  22 帧是 7 × 405 = **2835**（不是 2430）。
- **错误二（层级）**：`4·H·N²·D` 是**单层**的注意力 FLOP，要乘 33 层才是每步。
  两个错误叠加，一度得出「注意力只占 0.4%」和「占 12%」两个都不对的数字。
- **正确值**：22 帧注意力 7.6 TFLOP/步（占 FFN 的 6.7%）；56 帧 44.9 TFLOP/步（占 16%）。
- **教训**：FLOP 口径必须和**逐算子实测占比**对账（实测注意力时间占比 14–18%，
  说明它的单位 FLOP 比 GEMM 贵，不是 FLOP 占比高）。

### 8.2 `ggml_nbytes(node)` 只是输出字节

- **现象**：用输出字节算「带宽」，CONCAT/GLU 这类多输入算子被低估 2–5 倍。
- **修正**：CONCAT 读 N 份、GLU 读 2 份，带宽估计要用实际访存而非输出大小。
  不过 39→56 帧 20–40 倍的差距远超这个误差，结论不受影响。

### 8.3 独立 kernel 快 ≠ 端到端变快

- **现象**：自写注意力 kernel 微基准做到 **8.76 TFLOPS**（链路里约 2.5），快 3 倍，
  但接进链路的预期收益只有 **5–10%**，因为注意力只占 12–18% 的时间。
- **教训**：先量占比，再决定要不要写 kernel。

### 8.4 CMake 的 `-DXXX=ON` 不等于编译宏

- **现象**：`-DGGML_PROFILE_OPS=ON` 配置成功，但 `#ifdef GGML_PROFILE_OPS` 不生效，剖析无输出。
- **根因**：除非 CMakeLists 显式 `add_compile_definitions`，它只是个缓存变量。
- **解决**：临时插桩直接在源码里 `#define`。

### 8.5 插桩代码的位置要早于使用点

- **现象**：计数器插在文件末尾，编译报一串 `use of undeclared identifier`。
- **解决**：放在 `#include` 之后、第一个使用点之前。

### 8.6 `--vae-tiling` 是帧数相关的，不是永远更快

- 22 帧：开 tiling **慢 2%**（81.6 vs 79.7 s）；56 帧：开 **快 4%**（361 vs 377 s）。
- 脚本已改成仅 `Frames ≥ 39` 时启用。`--vae-tile-size 512x512` 无额外收益。

### 8.7 `--max-vram` 不接受空设备名

- **现象**：`invalid --max-vram assignment '=10'`。
- **根因**：gen.ps1 里设备变量叫 `$dev`，误写成 `$Device`，拼出来是 `=10`；
  PowerShell 未定义变量在字符串里静默变空，不报错。

### 8.8 `hipcc` 的 `-Llib` 是相对路径

- **现象**：链接阶段找不到 `amdhip64` 等库。
- **解决**：绕开 hipcc，直接用 `clang.exe --hip-link -L<绝对路径>`。

### 8.9 「采样阶段快」不等于「端到端快」

- `--params-backend diffusion=ROCm0` 让采样快 6%，但总耗时 69 → 133 s（TE/VAE 失去 offload）。
- **教训**：任何优化都要看**总耗时**，不能只看 sampling 那一行。

### 8.10 剖析必须用「干净构建」做基线

已在 4.4 记过，这里再强调：本会话曾拿插桩版当基线，得出「自编译版 56 帧慢 24%」的错误结论，
清掉插桩后实际**持平**。所有对外数字必须来自无插桩的官方二进制。

### 8.11 用错误的问题形状做微基准，得出「调参到头」的错误结论

- **现象**：曾用 K=7168 测 q4_K GEMM，得到 15.8 TFLOPS，据此认为内核离地板很远；
  又在 sd.cpp 层做 J=128 的 A/B（同时改了 occupancy），噪声淹没收益，得出「调参到头」。
- **纠正**：从 GGUF 里读出真实形状是 `mlp.fc1 = [5376, 28672]`（K=**5376**，不是 7168）。
  用真实形状复测，J=128 带来 **+7.9% 内核 / -4.7% 端到端**。
- **教训**：
  1. 微基准的形状必须从模型文件里读，不能猜。
  2. 一次只改一个变量（当时同时改了 J 和 occupancy）。
  3. 配置表要检查是否有编译期上限（`static_assert(J <= 128)`），原厂表只到 64 说明没调满。

### 8.12 PowerShell 的 `[regex]::Replace` 第 4 个参数是 Options，不是次数

- **现象**：想「只替换第一处」，写了 `[regex]::Replace($t, $pattern, $repl, 1)`，
  结果 11 处全被替换（`1` 被当成 `RegexOptions.IgnoreCase`）。
- **解决**：用 `[regex]::Replace($t, $p, $r)` 配 `new Regex(..., RegexOptions.None)` 控制次数，
  或者干脆按行遍历手动插入。

### 8.13 页缓存状态会让 --mmap 的采样时间差 35%

- **现象**：同一配置测出 49.9 s 和 67.4 s 采样，差 35%。
- **根因**：--mmap 下模型权重靠页缓存；冷缓存时权重反复缺页重读（diffusion load 17.3 s vs 0.2 s）。
- **解决**：任何 A/B 前先跑一次热身；跨会话的数字不可比。

### 8.14 并行跑 GPU 基准 = 数据报废

- **现象**：三个子代理并行，共享一块 GPU，注意力 A/B 得到 366/448 s（正常 ~139 s），
  GEMM 扫描里 q4_k 从 25.4 掉到 6.9 TFLOPS。
- **教训**：GPU 基准必须串行。可以并行的只有「写代码 + 编译」。正确组合是
  「一个代理占用 GPU 测量，其余改代码/写文档」。

## 九、一句话总结

这个项目里，**靠调参拿到的最大收益是 2.8 倍（WDDM 那个坑）**，
其次是 TE 量化 15%、ROCm 后端 1.5–1.8 倍、LoRA 策略 7–8 倍。
所有「内核级微调」（tile/stream-K/occupancy）都在噪声内或者更差。
**瓶颈经常不在你以为的地方** —— 先做逐算子测量，再动手。
