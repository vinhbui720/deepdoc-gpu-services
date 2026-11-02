# deepdoc-gpu-services

Independent, offline-ready GPU HTTP service sources. No model weights, credentials, certificates, or deployment automation enabled. Code derived from local `deploy/station2-*` templates; no live service touched.

| Service | Endpoint | Runtime | Test |
|---|---|---|---|
| embedding | `POST /embed` | numpy, tokenizers, onnxruntime-gpu; CUDA | `cd embedding && python -m unittest` |
| reranker | `POST /rerank` | torch, transformers; CUDA | `cd reranker && python -m unittest` |
| NLI | `POST /nli`, `GET /health` | torch, transformers; CUDA | `cd nli && python -m unittest` |

Tests use no GPU/model; embedding test needs numpy. Runtime reads env variables documented in each `env.example`. Supply owner-only bearer token files (32+ bytes), TLS key/certificate, offline model assets, and trusted CUDA-enabled dependencies before starting. Bind only approved LAN interface; do not expose to public network. Reranker TLS is optional in source: set `RERANK_TLS_CERT` and `RERANK_TLS_KEY` for LAN deployment. No dependency lock or complete model artifact hashes available: see `models.json`; verify independently before production deployment. CI checks contracts only, not GPU inference. Deploy templates deliberately disabled; operator review required.

## Deployment gate

`.github/workflows/deploy-template.yml` tests PRs and main pushes. Main deploy waits for successful tests and a `self-hosted`, `linux`, `station2` runner plus protected `production` environment approval. Configure branch protection requiring checks before merge. Set environment variable `GPU_DEPLOY_ROOT` to pre-provisioned, runner-owned directory. Runner must be isolated from untrusted PR jobs, with narrowly scoped sudo/systemd restart permissions for `deepdoc-{embedding,reranker,nli}.service`; provision each `<root>/<service>/current` symlink, `releases/`, `models/<revision>/`, and `health-url` containing verified HTTPS health endpoint. Units must execute code via `current` and use existing owner-only env/token/TLS files outside release. Health endpoint must return 200 only when model ready; embedding/reranker currently lack such endpoint. Runner, units, environment protection, artifact pins, verified health endpoints, and dependency locks are **not provisioned here**. No live service touched.

Deployment deliberately fails before switch: embedding revision absent; reranker hashes absent. Fill exact 40-hex revisions and complete SHA-256 manifest, provision independently verified offline artifacts, and add authenticated readiness endpoints before production. No automatic model download: incomplete manifest must not initiate network fetch. Release copies changed source to new SHA directory, keeps prior release as backup, switches symlink, restarts unit, checks health, and rolls back failed service on failure. Multi-service partial success requires operator review; do not enable until transactional policy agreed. Restrict who can push main and approve production: self-hosted runner executes repository code.
