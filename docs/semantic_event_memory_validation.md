# Semantic Event Memory Validation

## Scope

This validation checks EdgeLog's semantic event workflow:

```text
cheap trigger -> proposal -> verifier YES/NO/UNKNOWN -> semantic event memory -> search/daily summary
```

This validation keeps using the mock FINAL_ANSWER adapter so the semantic workflow can be tested without external services. Live RTX SmolVLM2 integration is validated separately in `docs/smolvlm2_fast_verifier_integration.md` and `serving/results/raw/smolvlm2_fast_verifier_validation.csv`.

The full real-mode stack is validated separately in `docs/edgelog_real_mode_validation.md` and `serving/results/raw/edgelog_real_mode_validation.csv`.

## Result Artifact

- `serving/results/raw/semantic_event_memory_validation.csv`
- Script: `scripts/validate_semantic_event_memory.py`

## Cases

| Case | Status | Notes |
| --- | --- | --- |
| motion/object_change proposal created | PASS | Cheap trigger creates proposal metadata. |
| person/ROI proposal created | PASS | Person in ROI creates proposal. |
| YOLO label proposal created | PASS | YOLO label is treated as proposal evidence, not final event. |
| verifier YES promotes proposal | PASS | Event becomes `verified` / `semantic_event`. |
| verifier NO rejects proposal | PASS | Rejected candidate is not counted as meaningful event. |
| verifier UNKNOWN marks needs review | PASS | Event is retained as unknown. |
| verifier failure does not block monitoring | PASS | Failure is recorded; workflow continues. |
| duplicate proposals suppressed | PASS | Cooldown prevents repeated event spam. |
| search returns verified event | PASS | Local JSONL search finds verified semantic event. |
| daily summary excludes rejected detections | PASS | Summary counts verified semantic events, not rejected object detections. |

## SmolVLM2 Status

SmolVLM2-256M is now available as a live RTX fast-verifier service when `verifier_backend=smolvlm2_fast` and the `127.0.0.1:8092` service is running.

Current evidence:

- `serving/docs/fast_vlm_verifier_benchmark.md`: compact SmolVLM2 models are sub-second on RTX but poor at strict JSON.
- `serving/docs/vlm_output_robustness.md`: simpler FINAL_ANSWER/final-line protocols are more suitable.
- `docs/smolvlm2_fast_verifier_integration.md`: live 256M service passed YES / NO / UNKNOWN, fallback, and EdgeLog promotion validation.

Conclusion: SmolVLM2 is now a real fast semantic verifier path on RTX, not a free-form describer and not a strict JSON backend.

## Gateway Impact

No core gateway routing policy was changed. This validation is demo/product workflow logic on top of the existing local/remote/reject infrastructure.

## Live Stack Note

`python3 demo/check_interactive_stack.py` was run during this pass. The Jetson Gateway was reachable and the local backend was available, while the gateway reported remote LLM/VLM unavailable through the current tunnel state. This does not affect the mock verifier workflow validation, but a live remote semantic demo should restart/check the interactive stack first.
