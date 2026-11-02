# deepdoc-gpu-services

Independent, offline-ready GPU HTTP service sources. No model weights, credentials, certificates, or deployment automation enabled. Code derived from local `deploy/station2-*` templates; no live service touched.

| Service | Endpoint | Runtime | Test |
|---|---|---|---|
| embedding | `POST /embed` | numpy, tokenizers, onnxruntime-gpu; CUDA | `cd embedding && python -m unittest` |
| reranker | `POST /rerank` | torch, transformers; CUDA | `cd reranker && python -m unittest` |
| NLI | `POST /nli`, `GET /ready` | torch, transformers; CUDA | `cd nli && python -m unittest` |

Tests use no GPU/model; embedding test needs numpy. Runtime reads env variables documented in each `env.example`. Supply owner-only bearer token files (32+ bytes), TLS key/certificate, offline model assets, and trusted CUDA-enabled dependencies before starting. Bind only approved LAN interface; do not expose to public network. Reranker requires `RERANK_TLS_CERT` and `RERANK_TLS_KEY` at startup. No dependency lock or complete model artifact hashes available: see `models.json`; verify independently before production deployment. CI checks contracts only, not GPU inference. Deploy templates deliberately disabled; operator review required.

## Deployment gate

`.github/workflows/deploy-template.yml` runs tests on PRs, main pushes, and manual dispatch. It has **no deploy job**. `deploy/release.py` exits nonzero without changing services. Do not enable automatic deployment or run on station2 yet.

Blockers: no validated isolated unprivileged station2 runner or pre-provisioned user-level release units (`systemctl --user`, not system-level); no verified release root/symlinks; embedding immutable revision missing, reranker artifact hashes missing, and no validated pinned model download/provisioning path. `/ready` now requires bearer auth and checks CUDA presence for each service, but does not execute inference: add authenticated functional probes before release switching. Dependency locks, certificate pinning, and protected production approval remain unverified. Keep credentials, model weights, and TLS files outside releases. No live service touched.

## Proposed station2 release contract (not enabled)

One dedicated runner, registered only for this repo, tagged `station2-gpu`, unprivileged and unable to write system units or other service directories. Require protected environment approval for manual `workflow_dispatch` with exactly one service (`embedding`, `reranker`, or `nli`) and an immutable commit SHA. Never deploy from a PR. Serialize deployments per service; do not cancel a running release. Pin actions by SHA, disable persisted checkout credentials, and grant workflow only `contents: read`. Build and test on runner with offline dependencies and model assets verified against `models.json`; reject incomplete revision or hashes. Runner must not fetch mutable model aliases.

Pre-provision separate user-level units and per-service roots: `releases/<service>/<commit>/`, `current/<service>` symlink, and an owner-only token file plus TLS key/cert outside release tree. Stage immutable release in same filesystem; never overwrite current release. Under a per-service lock, remember previous symlink target, atomically replace only selected symlink, restart only selected user unit, then run a bearer-authenticated functional inference probe through pinned TLS certificate. On any failure restore previous symlink atomically, restart selected unit and verify previous response; fail closed and alert operator if rollback probe fails. Preserve prior release until explicit cleanup. No shared restart, no cross-service switch, no automatic production rollout. Do not enable until runner identity, unit mappings, port uniqueness, model pins, dependency locks, and rollback behavior are verified on station2.
