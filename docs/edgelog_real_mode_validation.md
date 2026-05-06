# EdgeLog Real-Mode Validation

## Scope

This validation checks the live EdgeLog stack after the review branch was created. It verifies that the product loop is operational with real services where available:

```text
Jetson cheap trigger -> RTX SmolVLM2 verifier -> RTX Gemma describer -> event search / daily summary
```

Artifact:

- `serving/results/raw/edgelog_real_mode_validation.csv`

## Stack Health

The stack health case passed with:

- Jetson Gateway reachable at `http://192.168.1.102:8000`.
- Jetson local backend available.
- RTX remote LLM available through the Jetson reverse tunnel.
- RTX Gemma VLM available through the Jetson reverse tunnel.
- RTX SmolVLM2 verifier ready on `127.0.0.1:8092`.
- `vision_remote_is_mock=false`.

The remote LLM/VLM tunnel had been stale before this pass. It was restored using a temporary reverse tunnel helper under `runtime_logs/`; those runtime files are ignored and are not part of the commit.

## Results

| Case | Source | Status | Evidence |
| --- | --- | --- | --- |
| health_full_stack | real stack health | PASS | gateway/local/remote/VLM/verifier all ready |
| cheap_trigger_real_camera | real camera | PASS | local YOLO TensorRT route, `15.22 ms` inference |
| smolvlm2_fast_verifier_real | sample keyframe + real SmolVLM2 | PASS | `smolvlm2_256m_final_line`, `267.15 ms`, parse success |
| slow_vlm_describer_real | sample keyframe + real Gemma VLM | PASS | route remote, `remote_is_mock=false`, `16322.17 ms` |
| text_llm_summary_real | real text LLM | PASS | route remote, `qwen35_4b_q4`, `538.9 ms` |
| event_search_real_store | temporary event store | PASS | verified event searchable, rejected event excluded |
| daily_summary_real_or_deterministic | temporary event store | PASS | deterministic counts correct |
| storage_retention_guard | temporary event store | PASS | ordinary frame not retained as long-term event |
| privacy_local_only_reject | policy path | PASS | semantic VQA local-only route rejected |
| graceful_backend_failure | simulated unused verifier port | PASS | event failure recorded without crash |

## Real vs Sample/Simulated Boundary

Real hardware/service checks:

- real Jetson Gateway health;
- real Jetson camera capture and YOLO TensorRT inference;
- real RTX SmolVLM2 verifier service;
- real RTX Gemma VLM service through Jetson route;
- real RTX text LLM route through Jetson gateway.

Sample or simulated validation:

- SmolVLM2 and Gemma semantic cases use the committed non-sensitive sample keyframe, not a private camera frame.
- Search, daily summary, retention, and graceful failure use a temporary event store so runtime history and private images are not committed.

## Interpretation

SmolVLM2 is now a live fast semantic verifier on RTX, not just benchmark evidence. It does not replace YOLO cheap triggers and does not replace Gemma descriptions. Gemma remains slow but valid for asynchronous event description. The live validation keeps EdgeLog's core product boundary intact: fast local proposal, fast semantic verification, slow optional description, searchable retained event memory.

## Known Limits

- SmolVLM2 is on RTX, not Jetson.
- Gemma VLM latency is still slow enough to remain asynchronous only.
- Clip storage is schema-ready but not a ring buffer yet.
- Search is JSONL keyword/filter search, not FTS or embedding retrieval.
- The temporary Paramiko tunnel used during validation should be replaced by the normal SSH reverse tunnel process for demos.
