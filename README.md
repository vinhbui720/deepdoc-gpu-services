# deepdoc-gpu-services

Independent, offline-ready GPU HTTP service sources. No model weights, credentials, certificates, or deployment automation enabled. Code derived from local `deploy/station2-*` templates; no live service touched.

| Service | Endpoint | Runtime | Test |
|---|---|---|---|
| embedding | `POST /embed` | numpy, tokenizers, onnxruntime-gpu; CUDA | `cd embedding && python -m unittest` |
| reranker | `POST /rerank` | torch, transformers; CUDA | `cd reranker && python -m unittest` |
| NLI | `POST /nli`, `GET /health` | torch, transformers; CUDA | `cd nli && python -m unittest` |

Tests use no GPU/model; embedding test needs numpy. Runtime reads env variables documented in each `env.example`. Supply owner-only bearer token files (32+ bytes), TLS key/certificate, offline model assets, and trusted CUDA-enabled dependencies before starting. Bind only approved LAN interface; do not expose to public network. Reranker TLS is optional in source: set `RERANK_TLS_CERT` and `RERANK_TLS_KEY` for LAN deployment. No dependency lock or complete model artifact hashes available: see `models.json`; verify independently before production deployment. CI checks contracts only, not GPU inference. Deploy templates deliberately disabled; operator review required.
