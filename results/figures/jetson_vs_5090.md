# Jetson Orin Nano vs RTX 5090

| Model | Quant | RTX 5090 decode tok/s | Jetson Orin Nano decode tok/s | Edge/Base decode | RTX 5090 prefill tok/s | Jetson Orin Nano prefill tok/s | Edge/Base prefill | RTX 5090 peak MB | Jetson Orin Nano peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 169.32 | 32.68 | 0.193 | 155.45 | 146.26 | 0.941 | 5192 | 3664 |
| Qwen3.5 0.8B | Q4_K_M | 389.38 | 57.56 | 0.148 | 450.08 | 396.76 | 0.882 | 3865 | 2616 |
