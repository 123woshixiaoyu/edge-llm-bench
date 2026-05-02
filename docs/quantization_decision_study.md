# Quantization Decision Study on Jetson Orin Nano

Date: 2026-05-02

## Engineering Question

On a Jetson Orin Nano 8GB, which model size and quantization format is the most reasonable engineering choice for edge LLM deployment?

This stage is not only about running `llama-quantize`. The goal is to connect quantization to deployment constraints: memory headroom, decode latency, power, thermals, and whether output quality remains usable.

## Device Constraints

- Device: Jetson Orin Nano 8GB
- Memory: 8GB shared system/GPU memory; practical deployment needs headroom for the OS, application code, context cache, and other processes.
- Power class: roughly 25W class edge device. In this benchmark the 0.8B runs stayed around 19-20W average max power.
- Temperature: measured max temperature was 67-68C for the 0.8B runs.
- Stability target: no OOM, full GPU offload where possible, and consistent completion across the fixed prompt set.
- Latency target: decode speed should be comfortably above interactive thresholds. In this study, Q4_K_M is the only 0.8B format above 50 tok/s on Jetson.

## Quantization Background

F16 keeps weights at half precision. It is the closest local GGUF reference to the original model, but it uses the most memory bandwidth and storage.

Q8_0 stores most weights in 8-bit form. It is often a useful quality-preserving compression level: smaller and faster than F16, while usually close to F16 output behavior.

Q4_K_M is a mixed 4-bit quantization format. It is more aggressive: much smaller, generally lower memory pressure, and often faster on memory-bound edge devices. The trade-off is that small wording, reasoning, or factual mistakes become more likely.

For this project, quantization is useful only if it improves the deployment decision. The real engineering question is whether the smaller model file and lower memory bandwidth translate into better Jetson behavior without unacceptable output degradation.

## Experiment Matrix

Primary PTQ model: `Qwen/Qwen3.5-0.8B`, confirmed from the existing GGUF embedded base-model metadata and reproduced from the official Hugging Face source model.

| Artifact | Path | Size GiB |
|---|---|---:|
| F16 GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-f16.gguf` | 1.413 |
| Q8_0 GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q8_0.gguf` | 0.756 |
| Q4_K_M GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q4_K_M.gguf` | 0.493 |

Devices:

| Device | Backend | Monitor | Prompt set |
|---|---|---|---|
| Jetson Orin Nano 8GB | llama.cpp CUDA | `tegrastats` | 10 prompts |
| RTX 5090 Laptop 24GB / WSL2 | llama.cpp CUDA | `nvidia-smi` | 10 prompts |

Output files:

- `results/raw/qwen35_08b_quant_decision_jetson.csv`
- `results/raw/qwen35_08b_quant_decision_jetson_summary_by_model.csv`
- `results/raw/qwen35_08b_quant_decision_5090.csv`
- `results/raw/qwen35_08b_quant_decision_5090_summary_by_model.csv`
- `results/raw/qwen35_08b_quant_quality_samples.jsonl`
- `results/raw/qwen35_08b_quant_quality_scores.csv`
- `results/raw/qwen35_08b_quant_quality_long_samples.jsonl`
- `results/raw/qwen35_08b_quant_quality_long_scores.csv`
- `results/raw/gemma4_e2b_q8_full_jetson.csv`
- `results/raw/gemma4_e2b_q8_full_jetson_summary_by_model.csv`

## Jetson Results

All three 0.8B formats ran successfully on Jetson with full GPU offload.

| Quant | Cases | Failures | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C | Avg load ms | GPU layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F16 | 10 | 0 | 246.60 | 24.98 | 3677 | 18.93 | 67 | 213.82 | 25/25 |
| Q8_0 | 10 | 0 | 405.13 | 47.10 | 2953 | 19.69 | 67 | 126.32 | 25/25 |
| Q4_K_M | 10 | 0 | 406.99 | 56.54 | 2703 | 20.20 | 68 | 124.84 | 25/25 |

Key Jetson deltas:

| Comparison | Size change | Decode speed | Peak memory | Interpretation |
|---|---:|---:|---:|---|
| Q8_0 vs F16 | 1.413 GiB -> 0.756 GiB | 24.98 -> 47.10 tok/s | 3677 -> 2953 MB | Q8 is a large practical gain over F16. |
| Q4_K_M vs F16 | 1.413 GiB -> 0.493 GiB | 24.98 -> 56.54 tok/s | 3677 -> 2703 MB | Q4 gives the best Jetson deployment profile. |
| Q4_K_M vs Q8_0 | 0.756 GiB -> 0.493 GiB | 47.10 -> 56.54 tok/s | 2953 -> 2703 MB | Q4 is smaller and faster than Q8 in this run. |

## Power Efficiency

Power draw alone is not the right metric for an edge LLM. The useful question is how much decoding or prompt processing the device delivers per watt. The efficiency metrics below use:

- `decode_tok_per_watt = avg_decode_tps / avg_max_gpu_power_w`
- `prompt_tok_per_watt = avg_prompt_tps / avg_max_gpu_power_w`

Jetson Orin Nano 8GB:

| Quant | Avg prompt tok/s | Avg decode tok/s | Avg max power W | Prompt tok/s/W | Decode tok/s/W |
|---|---:|---:|---:|---:|---:|
| F16 | 246.60 | 24.98 | 18.93 | 13.03 | 1.32 |
| Q8_0 | 405.13 | 47.10 | 19.69 | 20.58 | 2.39 |
| Q4_K_M | 406.99 | 56.54 | 20.20 | 20.15 | 2.80 |

The important point is not that Q4_K_M uses less power. In this run, Q4_K_M has slightly higher average max power than F16. The engineering win is that Q4_K_M has similar power draw but much higher useful throughput per watt: decode efficiency improves from 1.32 tok/s/W on F16 to 2.80 tok/s/W on Q4_K_M.

RTX 5090 control:

| Quant | Avg prompt tok/s | Avg decode tok/s | Avg max power W | Prompt tok/s/W | Decode tok/s/W |
|---|---:|---:|---:|---:|---:|
| F16 | 410.07 | 234.01 | 105.20 | 3.90 | 2.22 |
| Q8_0 | 619.16 | 315.88 | 89.05 | 6.95 | 3.55 |
| Q4_K_M | 534.88 | 354.49 | 95.96 | 5.57 | 3.69 |

RTX also benefits from quantization, but the Jetson result is more deployment-relevant: Q4_K_M turns almost the same power envelope into a much larger amount of usable decode throughput.

F16 can run on the Jetson for this 0.8B model, so the decision is not simply "F16 OOM." The stronger conclusion is that F16 is not a good default: it costs about 1GB more peak memory than Q4_K_M and gives less than half the Q4 decode speed.

Q8_0 is useful as a quality-preserving middle point, but on Jetson it does not beat Q4_K_M on speed or memory. Its main reason to exist is quality conservatism.

Q4_K_M is the strongest default for this device because it gives the smallest file, lowest measured peak memory, fastest decode, and no observed failures.

## RTX 5090 Control

The same model matrix was also run on the RTX 5090 Laptop. This confirms that quantization affects both devices, but the Jetson result is more decision-relevant because memory bandwidth and shared memory constraints dominate edge deployment.

| Quant | Cases | Failures | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C | Avg load ms | GPU layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F16 | 10 | 0 | 410.07 | 234.01 | 5107 | 105.20 | 58 | 136.14 | 25/25 |
| Q8_0 | 10 | 0 | 619.16 | 315.88 | 4263 | 89.05 | 57 | 86.95 | 25/25 |
| Q4_K_M | 10 | 0 | 534.88 | 354.49 | 3952 | 95.96 | 56 | 104.21 | 25/25 |

On RTX, Q4_K_M is still fastest for decode, but the system has enough VRAM that memory pressure is not the central constraint. On Jetson, the same reduction directly translates into deployment headroom and a much larger relative decode-speed gain.

## Quality Evaluation

Quality was evaluated as a deployment sanity check, not as an academic benchmark. The goal is to catch obvious regressions from quantization before making a deployment recommendation.

Five representative prompts were used:

- Chinese QA: `zh_qa_edge_ai`
- English technical explanation: `en_qa_quant`
- Code generation: `code_python_csv`
- Summary: `summary_en`
- Reasoning/planning: `reasoning_ops`

Scores are coarse manual ratings:

- `instruction_following`: 0 bad, 1 partial, 2 good
- `completeness`: 0 bad, 1 partial/truncated, 2 complete
- `obvious_hallucination`: yes/no
- `code_runs_or_plausible`: yes/no/not_applicable

### Short-Generation Smoke

The first quality pass reused the original short benchmark generation lengths. It was useful as a smoke check, but many answers were truncated.

| Quant | Samples | Avg instruction following | Avg completeness | Obvious hallucinations |
|---|---:|---:|---:|---:|
| F16 | 5 | 1.60 | 1.20 | 1 |
| Q8_0 | 5 | 1.40 | 1.20 | 1 |
| Q4_K_M | 5 | 1.40 | 1.00 | 1 |

The short generation length makes these scores rough. They mainly show that all three formats can produce relevant outputs, while Q4_K_M already showed one clear technical wording issue in the quantization explanation.

### Long-Generation Sanity Check

To make the quality check fairer, the same 5 prompts were rerun on Jetson with `temperature=0` and `max_tokens=256`.

Output files:

- `results/raw/qwen35_08b_quant_quality_long_samples.jsonl`
- `results/raw/qwen35_08b_quant_quality_long_scores.csv`

| Quant | Samples | Avg instruction following | Avg completeness | Obvious hallucinations |
|---|---:|---:|---:|---:|
| F16 | 5 | 1.20 | 1.00 | 3 |
| Q8_0 | 5 | 1.20 | 1.00 | 3 |
| Q4_K_M | 5 | 0.80 | 0.80 | 4 |

This longer pass is still not a formal quality benchmark, but it gives a clearer deployment sanity check than the short generation. The main findings are:

- Q4_K_M is still usable for simple explanation and planning prompts, but it is less stable than Q8_0/F16 in this small sample.
- Q4_K_M produced the weakest code-generation sample, drifting into irrelevant repeated imports instead of a clean CSV append function.
- Q8_0 looked more conservative than Q4_K_M in the code sample, but it still made factual mistakes in technical explanation and planning.
- F16 was not clearly superior in this small sample. It also produced technical inaccuracies, so higher precision alone did not guarantee correctness for this small model.
- No result here justifies changing the deployment recommendation away from Q4_K_M, but it does justify keeping Q8_0 as the quality-conservative fallback.

## 4B Representative Jetson Run

After completing the 0.8B decision matrix, Qwen3.5 4B PTQ Q4_K_M was tested on Jetson with the full 10-prompt representative set. This upgrades the earlier 3-prompt smoke test into a more useful feasibility run for the quality-oriented candidate model.

| Model | Quant | Cases | Failures | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C | GPU layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B PTQ | Q4_K_M | 10 | 0 | 2.523 | 196.35 | 16.86 | 4549 | 23.91 | 72 | 33/33 |

Full GPU offload succeeded for all runs: 33/33 layers. There were no OOMs, timeouts, or failed prompts. Peak memory reached 4549 MB, leaving less headroom than the 0.8B Q4_K_M run but still fitting on the Orin Nano 8GB under the current benchmark settings.

Comparison with 0.8B Q4_K_M on Jetson:

| Model | Quant | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C | GPU layers |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B PTQ | Q4_K_M | 406.99 | 56.54 | 2703 | 20.20 | 68 | 25/25 |
| Qwen3.5 4B PTQ | Q4_K_M | 196.35 | 16.86 | 4549 | 23.91 | 72 | 33/33 |

The 4B Q4 model is feasible on Jetson Orin Nano 8GB as a quality-oriented candidate, but it should not replace Qwen3.5 0.8B Q4_K_M as the low-latency default unless the application can tolerate much lower decode throughput. Compared with 0.8B Q4_K_M, 4B Q4_K_M uses about 1.85 GB more peak memory, draws somewhat more power, reaches a higher max temperature, and decodes at roughly 30% of the 0.8B Q4_K_M speed.

4B F16 and Q8 were not run on Jetson in this stage because this run is specifically scoped to Q4_K_M feasibility. Their local RTX results and file sizes suggest they are less appropriate for the 8GB Jetson default path.

## Gemma 4 E2B Jetson Comparison

Gemma 4 is included as a non-Qwen medium model family and as a bridge toward future VLM work. The current local Gemma 4 setup uses prequantized text GGUF files plus a separate `mmproj-F16.gguf`; it is therefore a deployment comparison line, not a locally reproduced PTQ line like Qwen3.5.

Gemma 4 E2B Q4_K_M was already part of the Jetson representative run. Gemma 4 E2B Q8_0 was then added on Jetson with the same 10-prompt representative set.

| Model | Quant | Cases | Failures | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Decode tok/s/W | Max temp C | GPU layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 10 | 0 | 2.893 | 146.26 | 32.68 | 3664 | 21.90 | 1.49 | 68 | 36/36 |
| Gemma 4 E2B it | Q8_0 | 10 | 0 | 4.702 | 98.62 | 20.21 | 4485 | 20.58 | 0.98 | 67 | 36/36 |

Gemma 4 E2B Q8_0 can run on Jetson Orin Nano 8GB with full GPU offload and zero failures, so it is feasible. However, it is not a good default: compared with Gemma 4 E2B Q4_K_M, it is about 1.81 GiB larger on disk, uses about 821 MB more peak memory, and decodes at only about 62% of Q4_K_M throughput. Its measured max power is slightly lower, but useful decode throughput per watt is also lower.

This makes Gemma 4 E2B Q4_K_M the better Gemma-side deployment candidate on Jetson. It is slower than Qwen3.5 0.8B Q4_K_M, but it is a more realistic medium-model candidate and keeps Gemma 4 in the project as the path toward future multimodal camera experiments.

## Decision

Recommended Jetson default:

**Qwen3.5 0.8B Q4_K_M**

Reason:

- Fastest Jetson decode among the 0.8B formats: 56.54 tok/s.
- Smallest model file: 0.493 GiB.
- Lowest measured Jetson peak memory: 2703 MB.
- Best Jetson decode efficiency among the 0.8B formats: 2.80 tok/s/W.
- Full GPU offload: 25/25 layers.
- No failures across 10 prompts.
- Similar power/thermal envelope to F16 and Q8, but much better useful throughput and throughput per watt.

Alternative:

**Qwen3.5 0.8B Q8_0** if quality conservatism matters more than memory and decode speed. Q8_0 is still a strong improvement over F16 and looked more stable than Q4_K_M in the code-generation long sample, but in the performance run Q4_K_M was smaller, faster, lower-memory, and more decode-efficient.

Not recommended as Jetson default:

**Qwen3.5 0.8B F16**. It can run, but it is not efficient on Orin Nano: 24.98 tok/s decode, 1.32 decode tok/s/W, 3677 MB peak memory, and the largest file size. F16 is best kept as a local reference artifact, not the default edge deployment format.

Conditional option:

**Qwen3.5 4B Q4_K_M** is feasible for the full 10-prompt Jetson representative run, but at 16.86 tok/s decode it is a quality-over-latency candidate, not the main low-latency recommendation.

Non-Qwen medium-model option:

**Gemma 4 E2B Q4_K_M** is the preferred Gemma-side Jetson candidate. It runs the full representative set with 32.68 tok/s decode and 3664 MB peak memory. Gemma 4 E2B Q8_0 is feasible, but its larger file size, higher peak memory, and lower decode throughput make it less attractive for the 8GB Jetson default path.

## Engineering Interpretation

Jetson shows quantization value more clearly than RTX because it is constrained in the dimensions that matter for edge deployment: shared 8GB memory, lower memory bandwidth, lower power budget, and tighter thermal headroom. On RTX, all three 0.8B formats fit comfortably and run fast enough that the decision can look cosmetic. On Jetson, the same quantization step changes the product behavior: F16 is usable but slow, Q8 nearly doubles decode speed, and Q4 more than doubles decode speed while reducing peak memory.

Just running PTQ is not enough because a quantized file by itself does not answer the deployment question. A model can be successfully quantized but still be too slow, too memory-heavy, too power-inefficient, or too unstable in output quality. The engineering decision has to join four views:

- Memory: whether the model leaves enough headroom for the device and application.
- Latency: whether decode speed is interactive enough.
- Power efficiency: whether the device gets useful throughput per watt, not merely lower watts.
- Quality: whether the quantized model still follows instructions and avoids obvious regressions.

The final recommendation is therefore not "Q4 because smaller" or "Q8 because higher precision." It is Q4_K_M as the Jetson default because the measured system-level trade-off is best, with Q8_0 kept as a conservative fallback when output stability matters more than speed and memory.

The Gemma 4 E2B result shows that this conclusion is not only a Qwen-specific artifact. On a medium non-Qwen model, Q4_K_M again gives the better edge deployment profile than Q8_0. Gemma also remains valuable because it connects the current text deployment benchmark to a future VLM path with `mmproj` and camera input.

## Lessons Learned

- Quantization solved a real Jetson deployment problem: it reduced file size, lowered memory pressure, improved decode tok/s/W, and more than doubled decode speed from F16 to Q4_K_M for Qwen3.5 0.8B.
- Q8_0 is not automatically the best edge compromise. On this Jetson run, Q4_K_M was both faster and smaller, while remaining usable in the rough quality check.
- RTX 5090 results are useful as a control, but they hide the urgency of quantization because the GPU has enough VRAM and bandwidth. Jetson makes the trade-off visible.
- Running PTQ alone is not enough. The engineering decision requires the full loop: source model, converted GGUF, quantized artifacts, device benchmark, power/thermal/memory data, quality samples, and an explicit recommendation.
