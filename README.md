# deepdoc-gpu-services

Independent, offline-ready GPU HTTP service sources. No model weights, credentials, certificates, or deployment automation enabled. Code derived from local `deploy/station2-*` templates; no live service touched.

| Service | Endpoint | Runtime | Test |
|---|---|---|---|
| embedding | `POST /embed` | numpy, tokenizers, onnxruntime-gpu; CUDA | `cd embedding && python -m unittest` |
| reranker | `POST /rerank` | torch, transformers; CUDA | `cd reranker && python -m unittest` |
| NLI | `POST /nli`, `GET /health` | torch, transformers; CUDA | `cd nli && python -m unittest` |

Tests use no GPU/model; embedding test needs numpy. Runtime reads env variables documented in each `env.example`. Supply owner-only bearer token files (32+ bytes), TLS key/certificate, offline model assets, and trusted CUDA-enabled dependencies before starting. Bind only approved LAN interface; do not expose to public network. Reranker TLS is optional in source: set `RERANK_TLS_CERT` and `RERANK_TLS_KEY` for LAN deployment. No dependency lock or complete model artifact hashes available: see `models.json`; verify independently before production deployment. CI checks contracts only, not GPU inference. Deploy templates deliberately disabled; operator review required.

## Deployment gate

`.github/workflows/deploy-template.yml` runs tests on PRs, main pushes, and manual dispatch. It has **no deploy job**. `deploy/release.py` exits nonzero without changing services. Do not enable automatic deployment or run on station2 yet.

Blockers: no validated isolated unprivileged station2 runner or pre-provisioned user-level release units (`systemctl --user`, not system-level); no verified release root/symlinks or transactional multi-service rollback policy; embedding immutable revision missing, reranker artifact hashes missing, and no validated pinned model download/provisioning path; embedding/reranker lack functional readiness endpoints, while all health checks need bearer authentication and certificate pinning. Dependency locks and protected production approval also remain unverified. Keep credentials, model weights, and TLS files outside releases. Confirm these prerequisites and add tests for authenticated functional checks and rollback before writing/enabling a release switch. No live service touched.
