# Model / Backend Selection Scorecard

## Purpose

The project is a Jetson-first, constraint-aware edge AI inference gateway. Model choice should therefore be an engineering decision, not a single leaderboard sorted by parameter count or speed.

The scorecard uses two layers:

1. hard constraints that remove candidates which do not satisfy deployment requirements;
2. profile-specific weighted scoring that reflects different product goals.

This matters because the best backend depends on the request. A private short QA task, a quality-oriented text task, a fast object detection task, and a semantic visual reasoning task should not be forced into one absolute ranking.

## Inputs

Candidate data lives in:

```text
results/raw/model_selection_candidates.csv
```

Profile weights and hard constraints live in:

```text
configs/model_selection_profiles.yaml
```

Scores are generated with:

```bash
python3 scripts/score_model_candidates.py \
  --candidates results/raw/model_selection_candidates.csv \
  --profiles configs/model_selection_profiles.yaml \
  --out results/raw/model_selection_scores.csv
```

The candidate table uses existing Project 1/2/3 results. No new benchmark was run for this step.

## Candidate Families

The scorecard uses the same schema for LLM, CV, and VLM candidates:

- text local LLM: Jetson `llama.cpp` models for default/private text tasks;
- text fallback LLM: stronger local or remote text models for quality-oriented requests;
- local CV: Jetson OpenCV / ONNXRuntime / TensorRT object detection runtimes;
- remote VLM: RTX visual semantic backend for VQA and scene description.

Some fields are naturally empty for some families. For example, `decode_tok_s` is meaningful for LLMs but not CV, while `cv_total_latency_ms` is meaningful for CV but not LLMs. Missing non-critical fields receive a neutral score and are listed in `score_notes`.

## Profiles

### text_local_default

Goal: choose the default Jetson text model.

Hard constraints:

- `failures == 0`;
- Jetson local backend;
- `peak_memory_mb <= 6000`;
- `decode_tok_s >= 15`;
- must support `local_only` privacy.

Weights emphasize latency, memory, power efficiency, and stability.

### text_quality_fallback

Goal: choose the stronger quality-oriented text fallback.

Hard constraints:

- `failures == 0`;
- `decode_tok_s >= 10`;
- manual quality heuristic must be at least `4.0 / 5`.

This profile intentionally excludes 0.8B models even if they are efficient, because they are default-path candidates rather than quality fallback candidates.

### vision_local_fast_path

Goal: choose the fast local Jetson CV runtime for detect/classify tasks.

Hard constraints:

- `failures == 0`;
- must support `local_only` privacy;
- must have either `cv_total_latency_ms` or `cv_inference_latency_ms`.

Weights emphasize latency, stability, integration status, and rough local detection quality.

### vision_remote_semantic_backend

Goal: choose the remote semantic visual backend for VQA and scene description.

Hard constraints:

- `failures == 0`;
- `remote_is_mock == false`;
- must support `allow_remote`;
- must have `vlm_latency_ms`.

Weights emphasize quality, integration status, stability, and latency.

## Current Recommendations

| Profile | Recommendation | Score | Why |
|---|---|---:|---|
| `text_local_default` | Qwen3.5 0.8B `Q4_K_M` on Jetson | 90.60 | Highest decode throughput, lowest memory, best tok/s/W among Jetson local text candidates, and already integrated as the real local backend. |
| `text_quality_fallback` | Qwen3.5 4B `Q4_K_M` on RTX | 77.42 | Stronger 4B quality candidate with the best score after the quality hard gate; `Q8_0` is the conservative quality alternative but costs more memory and power. |
| `vision_local_fast_path` | YOLOv8n TensorRT FP16 on Jetson | 83.20 | Lowest measured local CV latency: `14.54 ms` inference and `28.98 ms` total, with 30/30 stable detections. |
| `vision_remote_semantic_backend` | Gemma 4 E2B-it `Q4_K_M` + `mmproj-F16` on RTX | 82.40 | Only current real remote VLM backend; usable for semantic visual tasks, but latency is high at about `19.48 s` per remote VLM call. |

Full score output:

```text
results/raw/model_selection_scores.csv
```

## Interpretation

The scorecard makes the project policy explicit:

- the Jetson default is chosen for stable low-latency private inference, not maximum raw model quality;
- the text fallback is allowed to spend more remote compute for quality;
- local CV is evaluated as an edge fast path, so runtime latency dominates;
- remote VLM is evaluated as semantic capability, so it is allowed to be slow but must be real and non-mock.

This also explains why the router should keep multiple backends. A single "best" model is not enough for a heterogeneous edge gateway.

## Limitations

- `quality_score_manual` is a manual heuristic from smoke-level quality checks, not an academic benchmark.
- CV candidates use different label sets and weights. MobileNet-SSD, SSD-MobileNetV1, and YOLOv8n are compared as deployment/runtime candidates, not as a direct accuracy contest.
- The remote VLM profile currently has one real candidate, so it validates availability and latency more than comparative model choice.
- Power and memory are missing for some CV/VLM candidates, so those sub-scores use neutral values and are marked in `score_notes`.
- Future scorecard rows should add Qwen3-VL or another VLM, YOLO INT8, calibration results, and a more systematic quality evaluation.

## Engineering Conclusion

The scorecard preserves the current project decisions but makes their assumptions reviewable:

- Jetson text default: Qwen3.5 0.8B `Q4_K_M`.
- Text quality fallback: Qwen3.5 4B `Q4_K_M` on RTX, with Qwen3.5 4B `Q8_0` as a conservative but heavier alternative.
- Jetson local CV fast path: YOLOv8n TensorRT FP16 for the optimized path; MobileNet-SSD remains the already-integrated v0.5 router baseline until the router policy is intentionally updated.
- Remote semantic vision: Gemma 4 E2B-it `Q4_K_M` + `mmproj-F16`, real but latency-heavy.

The score is not an absolute truth. It is a compact way to make deployment preferences explicit and reproducible.
