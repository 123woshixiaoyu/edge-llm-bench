# SmolVLM2 Output Robustness Test

## Goal

The previous fast VLM verifier benchmark showed that SmolVLM2 can run much faster than the current Gemma remote VLM path, but strict JSON validity was `0/36`. This follow-up asks whether that was only a formatting problem or whether the models are not reliable enough for semantic verifier use.

This test does not connect SmolVLM2 to the router or UI. It only evaluates output protocols.

## Setup

Runtime:

- Platform: `rtx_5090_wsl`
- Environment: isolated `.venv-vlm`
- Models:
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  - `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
  - `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Images: three committed public sample images from `results/figures/`
- Tasks:
  - lipstick presence
  - person falling or lying down
  - abnormal / attention-needed scene
  - general monitoring-frame description

Command:

```bash
source .venv-vlm/bin/activate
python serving/scripts/benchmark_vlm_output_robustness.py \
  --models smolvlm2_500m,smolvlm2_256m,smolvlm2_2b \
  --platform-label rtx_5090_wsl \
  --max-new-tokens 96 \
  --timeout-s 240 \
  --out serving/results/raw/vlm_output_robustness_benchmark.csv \
  --summary-out serving/results/raw/vlm_output_robustness_summary.csv
```

Artifacts:

- `serving/results/raw/vlm_output_robustness_benchmark.csv`
- `serving/results/raw/vlm_output_robustness_summary.csv`

## Prompt Protocols

| Variant | Purpose |
| --- | --- |
| `strict_json` | Original JSON-only instruction. |
| `strict_json_fewshot` | Adds one tiny JSON example before the real question. |
| `yes_no_token` | Asks for exactly `YES`, `NO`, or `UNKNOWN`. |
| `final_line` | Asks for `FINAL_ANSWER: YES|NO|UNKNOWN` on the final line. |
| `retry_strict` | Runs strict JSON first, then retries once with a stronger machine-parseable JSON prompt if JSON parsing fails. |

Parsing is intentionally separated from semantic correctness:

- `strict_json_valid_rate` checks exact `json.loads(output_text.strip())`.
- `relaxed_json_valid_rate` accepts fenced JSON or the first `{...}` substring.
- `yes_no_parse_rate` accepts unambiguous `YES` / `NO` / `UNKNOWN` answers.
- `semantic_match_rate` checks the two unambiguous public-image tasks, lipstick presence and person falling, where the expected answer is `no`.

## Results

| Model | Protocol | Avg latency | Parse success | Semantic match | Recommendation |
| --- | --- | ---: | ---: | ---: | --- |
| SmolVLM2-500M | `strict_json` | 435.09 ms | 0.75 | 0.00 | not suitable |
| SmolVLM2-500M | `strict_json_fewshot` | 301.13 ms | 0.75 | 1.00 | usable with human review |
| SmolVLM2-500M | `yes_no_token` | 273.79 ms | 1.00 | 0.667 | usable with retry / review |
| SmolVLM2-500M | `final_line` | 277.22 ms | 1.00 | 0.833 | event verifier candidate |
| SmolVLM2-500M | `retry_strict` | 1579.24 ms | 0.917 | 0.00 | not suitable |
| SmolVLM2-256M | `strict_json` | 360.60 ms | 0.50 | 0.50 | not suitable |
| SmolVLM2-256M | `strict_json_fewshot` | 611.65 ms | 0.833 | 0.333 | not suitable |
| SmolVLM2-256M | `yes_no_token` | 295.16 ms | 0.50 | 0.50 | not suitable |
| SmolVLM2-256M | `final_line` | 273.66 ms | 1.00 | 1.00 | event verifier candidate |
| SmolVLM2-256M | `retry_strict` | 1762.90 ms | 0.75 | 0.50 | not suitable |
| SmolVLM2-2.2B | `strict_json` | 528.58 ms | 0.75 | 1.00 | usable with human review |
| SmolVLM2-2.2B | `strict_json_fewshot` | 557.51 ms | 0.75 | 1.00 | usable with human review |
| SmolVLM2-2.2B | `yes_no_token` | 481.51 ms | 1.00 | 0.667 | usable with retry / review |
| SmolVLM2-2.2B | `final_line` | 569.05 ms | 1.00 | 1.00 | event verifier candidate |
| SmolVLM2-2.2B | `retry_strict` | 1480.05 ms | 1.00 | 0.50 | not suitable |

## Interpretation

Strict JSON validity being zero is not the whole story. The models can often answer in a simple parseable protocol, especially `final_line`. The best result for a fast verifier is:

- **SmolVLM2-256M + `final_line`**
- Avg latency: `273.66 ms`
- Overall parse success: `1.00`
- Semantic match on the unambiguous public-image yes/no checks: `1.00`

SmolVLM2-500M + `final_line` is also promising, but had one semantic mismatch in the limited expected-answer subset. SmolVLM2-2.2B is slower and uses substantially more memory, so it is less attractive unless quality proves better on a larger task set.

The retry strategy is not a clear win. It increases latency to roughly `1.5-1.8 s`, and the retry output can still be malformed or semantically wrong. For this model family, a simpler final-line protocol is more reliable than asking for strict JSON.

## Answers To The Product Questions

1. **Is JSON valid rate = 0 fatal?**  
   Not by itself. Strict JSON is a poor protocol for these models, but `final_line` can make output parseable.

2. **Which output protocol is most stable?**  
   `final_line` is the best protocol in this run. `yes_no_token` is parseable for some models, but less semantically reliable.

3. **Which SmolVLM2 size is best for a fast verifier?**  
   SmolVLM2-256M with `final_line` is the best first candidate because it is fastest and produced the best semantic match on the limited public-image checks. SmolVLM2-500M is the next candidate.

4. **Can SmolVLM2 be an automatic alert verifier?**  
   Not yet as a general verifier. It can become a candidate for narrow yes/no event checks after more task-specific validation. It should not directly drive safety-critical alerts from this small test.

5. **Does this replace YOLO?**  
   No. YOLO TensorRT remains the local monitoring fast path. SmolVLM2 is an event-level semantic helper, not a detector running at camera-loop speed.

6. **Should it be connected to event-triggered VLM review?**  
   It is worth a future optional experiment behind the event-triggered workflow, using `FINAL_ANSWER` parsing and cooldowns. Gemma should remain the quality fallback until compact-VLM semantic reliability is proven on a broader dataset.

## Recommended Next Step

Do not replace the current Gemma remote VLM yet. The practical next experiment is:

1. Keep YOLO TensorRT as the continuous local monitor.
2. Add an optional SmolVLM2-256M `final_line` verifier experiment for narrow candidate events only.
3. Build a small labeled validation set for target events such as person present, lying/fall, bottle, and specific custom objects.
4. Consider YOLO-pose or an open-vocabulary detector for event types that are primarily visual-detection problems.

