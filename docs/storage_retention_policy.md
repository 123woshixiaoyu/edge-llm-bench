# EdgeLog Storage Retention Policy

EdgeLog is local-first by design. The local cheap-trigger path can refresh snapshots repeatedly, but ordinary frames are not kept forever. Only proposals/events with semantic value are retained.

## What Is Saved

Long-term Event History records are saved when one of these conditions is true:

- the user explicitly saves the event;
- a cheap trigger creates a proposal that is verified, unknown, failed, or explicitly saved;
- a YOLO trigger rule matches and is promoted by the verifier;
- a local alert is generated;
- a VLM review is queued, running, done, or failed;
- privacy policy rejects a semantic review;
- a backend error or sample fallback happens;
- EdgeLog Daily Summary / Monitoring Assistant generates a summary.

Detection changes during ordinary auto refresh are not retained by themselves. They update latest-snapshot metadata unless the user saves the snapshot or a trigger/review/reject/error condition occurs.

Each saved event can include the EdgeLog schema fields: event rule, proposal evidence, verifier answer/reason, description, storage metadata, event type, start/end time, duration, ROI name, risk level, route/backend, latency, privacy mode, and retention metadata.

## What Is Not Saved

Auto-refresh frames that do not create a verified/unknown/failed proposal, reject, backend error, semantic review, or user save are not written to long-term Event History. They only update:

```text
runtime_data/events/latest_snapshot.jpg
```

That file is overwritten on each ordinary refresh. Per-frame monitoring stores cheap trigger metadata only. It does not generate scene descriptions or VLM text for every frame.

## Image Retention

Images are retained only for meaningful events. Metadata-only events are allowed when an image is unavailable or when privacy/storage policy avoids retaining the frame.

Default retention settings:

| Setting | Default |
| --- | ---: |
| Max events | 500 |
| Max retained images | 512 MB |
| Retention window | 7 days |
| Latest snapshot | overwritten |

Environment overrides:

```bash
MONITORING_MAX_EVENTS=500
MONITORING_MAX_IMAGES_MB=512
MONITORING_RETENTION_DAYS=7
```

After event append/update, the store cleans up expired events, trims the log to the newest events, deletes unreferenced images, and removes old images if the image budget is exceeded. If an event remains but its image was removed, it is marked:

```json
{"image_missing": true}
```

`latest_snapshot.jpg` is not deleted by cleanup.

## Privacy Fields

Each event includes retention/privacy metadata:

- `stored_image`
- `image_retention_reason`
- `sent_to_remote`
- `remote_backend`
- `privacy_mode`
- `retention_expires_at`
- `image_missing`
- `storage_policy_version`

For Event Review, the UI warns when an image may be sent to the RTX workstation for semantic VLM review. Privacy-safe semantic review is rejected and recorded with `sent_to_remote=false`.

## Runtime Location

Runtime history is intentionally ignored by git:

```text
runtime_data/events/events.jsonl
runtime_data/events/images/
runtime_data/events/latest_snapshot.jpg
```

Do not commit runtime event history, private camera frames, model weights, TensorRT engines, tokens, or `.env` files.
