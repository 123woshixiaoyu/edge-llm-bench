# Submission Checklist

## GitHub Visibility

The GitHub repository may be private. That is acceptable for development, but a private repository link will show `404` to reviewers who do not have access.

Before using the repo in a resume, portfolio, or application:

- Confirm the GitHub link is externally accessible, or prepare an alternate artifact.
- Recommended option: temporarily make the repo public before submitting.
- If keeping the repo private, write `demo/report available upon request` instead of linking an inaccessible repo.
- For a specific reviewer or interviewer, add them as a GitHub collaborator.
- A public sanitized version is also acceptable: include code, README, CSV summaries, and non-sensitive images; exclude models, engines, large logs, tokens, and private path details.

## Alternate Sharing Artifacts

Prepare one or more of:

- README export as PDF.
- Key docs as PDF.
- Demo dashboard screenshots.
- A short screen recording of the sample dashboard.
- Selected CSV summaries and figures.

## Do Not Share

Never paste or commit:

- SSH private keys, including blocks starting with `-----BEGIN OPENSSH PRIVATE KEY-----`.
- Hugging Face tokens or API keys.
- `.env` files.
- Model weights, GGUF files, ONNX files, TensorRT engines, or calibration caches.
- Large raw logs.

An SSH public key fingerprint such as a SHA256 fingerprint is normally not a private secret. The private key content itself is sensitive.

## Runtime Assets

Model files and runtime acceleration artifacts are intentionally outside git:

```text
/mnt/d/AI/Models
/home/rainbow/models
```

This is a design choice, not a missing commit. The repo should contain reproducible code, configs, docs, CSV summaries, and small evidence images, while large/licensed runtime assets remain local.
