# Edge LLM Quantized Deployment Benchmark Report

日期：2026-05-02  
当前阶段：RTX 5090 Laptop / WSL2 baseline 已完成，Jetson 尚未开始。

## 1. 项目目标

在 2 周内完成端侧 LLM 量化部署 benchmark，比较本地 GGUF 模型在不同硬件上的部署表现。当前范围只包含推理 benchmark，不包含微调、QAT、serving 平台或新增模型下载。

## 2. 今晚完成内容

- 在 `/mnt/d/AI/edge-llm-bench` 创建项目目录。
- 安装/确认依赖：`git`、`cmake`、`build-essential`、`python3`、`python3-venv`、`python3-pip`。
- clone 并 CUDA 编译 llama.cpp。
- 用 Gemma 4 E2B Q4 完成 smoke test，确认 CUDA backend 和 GPU offload 生效。
- 建立目录结构：`scripts/`、`prompts/`、`results/raw/`、`results/figures/`、`docs/`。
- 创建 10 条 benchmark prompts，覆盖中文问答、英文问答、代码、摘要、长上下文、推理。
- 编写 benchmark 脚本，输出逐 case CSV 和按模型聚合 summary CSV。
- 完成 RTX 5090 Laptop baseline：6 个模型 × 10 条 prompt，共 60 条记录，全部成功。

## 3. 环境记录

- OS：Windows + WSL2 Ubuntu-22.04
- GPU：NVIDIA GeForce RTX 5090 Laptop GPU
- VRAM：24463 MiB
- Driver：592.01
- CUDA toolkit：13.2，`nvcc` 位于 `/usr/local/cuda-13.2/bin/nvcc`
- llama.cpp commit：`b97ebdc`
- llama.cpp build：`cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` + `cmake --build build --config Release -j$(nproc)`
- Runtime：`llama-completion`
- 主要参数：`-ngl 99`、`-no-cnv`、`--no-display-prompt`、`--perf`、`--no-warmup`、`--temp 0`
- Context：大多数 prompt 使用 4096，long context prompt 使用 8192

注意：当前 load time 是 mmap + OS 文件缓存条件下的初步值，不等同于严格冷启动加载时间。

## 4. 产出文件

- Prompt 集：`prompts/benchmark_prompts.jsonl`
- Benchmark 脚本：`scripts/run_benchmark.py`
- Summary 脚本：`scripts/summarize_results.py`
- Smoke test 日志：`results/raw/smoke_gemma_e2b_q4.log`
- 逐 case 结果：`results/raw/5090_baseline.csv`
- 按模型汇总：`results/raw/5090_baseline_summary_by_model.csv`
- 运行日志：`results/raw/5090_baseline_run.log`
- 每个 case 原始日志：`results/raw/logs/`

## 5. RTX 5090 初步结果

| Model | Quant | Cases | Failures | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB | Avg max power W | Max temp C | GPU layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 10 | 0 | 2.893 | 155.45 | 169.32 | 5192 | 94.93 | 56 | 36/36 |
| Gemma 4 E2B it | Q8_0 | 10 | 0 | 4.702 | 165.46 | 128.22 | 6189 | 107.43 | 56 | 36/36 |
| Gemma 4 E4B it | Q4_K_M | 10 | 0 | 4.635 | 152.09 | 99.62 | 6581 | 100.22 | 55 | 43/43 |
| Qwen3.5 4B | Q4_K_M | 10 | 0 | 2.553 | 406.04 | 123.50 | 6222 | 108.91 | 56 | 33/33 |
| Qwen3.5 4B | Q8_0 | 10 | 0 | 4.175 | 649.71 | 97.82 | 7825 | 109.91 | 60 | 33/33 |
| Qwen3.5 0.8B | Q4_K_M | 10 | 0 | 0.496 | 450.08 | 389.38 | 3865 | 103.62 | 59 | 25/25 |

初步观察：

- 所有模型均完成 GPU offload，逐 case 日志中有 offloaded layer 记录。
- Q8 相比 Q4 通常占用更多显存，decode tok/s 更低。
- Qwen3.5 0.8B Q4 decode 速度最高，适合作为 Jetson smoke/baseline 快速验证模型。
- Qwen3.5 4B 的 prompt eval tok/s 明显高于 Gemma 系列，但 decode tok/s 与 Gemma E2B Q8 / E4B Q4 接近，需要在 Jetson 上复核。

## 6. 遇到的问题和处理

- `sudo` 需要密码：改用 WSL root 用户安装指定依赖。
- `apt-get update` 初次出现 DNS 警告，但 apt 使用已有索引并成功安装缺失包。
- `nvcc` 不在默认 PATH：确认 CUDA toolkit 已安装在 `/usr/local/cuda-13.2`，构建和脚本中显式设置 PATH。
- WSL 继承 Windows PATH 导致 bash 解析带空格/括号路径失败：构建时改用干净 Linux PATH。
- `llama-cli --no-conversation` 不被当前版本支持：改用 `llama-completion -no-cnv`。
- Gemma 自定义 chat template 在 conversation mode 下需要 `--jinja`，否则会报 custom template unsupported：benchmark 使用 completion mode 规避。
- WSL 偶发 `Failed to start the systemd user session` 警告：未阻塞编译或推理，暂记为环境噪声。

## 7. 明天 Jetson 待办

- 在 Jetson 上确认系统、JetPack/CUDA、`nvidia-smi` 或 `tegrastats` 可用情况。
- clone 或同步当前项目，不下载更多模型，不保存 Hugging Face token。
- 编译 Jetson 端 llama.cpp CUDA 版本；若 CUDA backend 失败，记录完整错误和已尝试步骤。
- 先用 Qwen3.5 0.8B Q4 做 smoke test，确认 GPU offload 和性能日志可用。
- 根据 Jetson 工具改造 GPU 采样：优先用 `tegrastats` 记录显存/功耗/温度。
- 先跑 3 个代表模型：Qwen3.5 0.8B Q4、Gemma 4 E2B Q4、Qwen3.5 4B Q4。
- 若时间和显存允许，再跑完整 6 模型 × 10 prompt。
- 生成 Jetson CSV 后，与 5090 summary 做对比表和图。

## 8. 明天需要确认的选择

Jetson 实验顺序有两种可选路线：

- 路线 A：先跑 3 个代表模型 × 10 prompt，确认脚本和指标稳定后再扩展到 6 个模型。
- 路线 B：直接跑 6 个模型 × 10 prompt，节省人工切换，但失败时定位成本更高。

建议路线 A。这样最符合端侧设备不确定性较高的现实，也能最快得到可展示结果。
