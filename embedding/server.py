"""Isolated LAN HTTPS multilingual-E5-small embedding service (ONNX Runtime, CUDA).

Same math as deeptutor/services/embedding/adapters/onnx_local.py: "query: " /
"passage: " prefix, mean pooling, L2 norm, 512 tokens max, 384 dims.
"""
import hmac
import json
import logging
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore

import numpy as np

MAX_BODY = 524288
MAX_BATCH = 64
MAX_CHARS = 8192
BUSY = BoundedSemaphore(1)
TOKEN = b""
SESSION = None
TOKENIZER = None


def load_token(path):
    with open(path, "rb") as f:
        token = f.read().strip()
    assert len(token) >= 32, "token too short"
    return token


def embed(texts, input_type):
    """Return L2-normalised 384-d vectors; prefix logic mirrors the CPU adapter."""
    prefix = "query: " if input_type == "search_query" else "passage: "
    encoded = TOKENIZER.encode_batch([prefix + t for t in texts])
    ids = np.zeros((len(encoded), min(512, max(len(e.ids) for e in encoded))), dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, e in enumerate(encoded):
        n = min(ids.shape[1], len(e.ids))
        ids[i, :n] = e.ids[:n]
        mask[i, :n] = 1
    out = SESSION.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
    vec = (out * mask[..., None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return vec.astype(np.float32).tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logging.info("%s %s", self.address_string(), format % args)

    def reply(self, status, data):
        raw = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/ready":
            return self.reply(404, {"error": "not found"})
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), b"Bearer " + TOKEN):
            return self.reply(401, {"error": "unauthorized"})
        ready = SESSION is not None and TOKENIZER is not None and SESSION.get_providers()[0] == "CUDAExecutionProvider"
        self.reply(200 if ready else 503, {"ready": ready})

    def do_POST(self):
        if self.path != "/embed":
            return self.reply(404, {"error": "not found"})
        auth = self.headers.get("Authorization", "")
        if not hmac.compare_digest(auth.encode(), b"Bearer " + TOKEN):
            return self.reply(401, {"error": "unauthorized"})
        length = self.headers.get("Content-Length", "")
        if not length.isdecimal() or not 0 < int(length) <= MAX_BODY:
            return self.reply(413, {"error": "body limit"})
        try:
            data = json.loads(self.rfile.read(int(length)))
            texts, kind = data["texts"], data.get("input_type", "search_document")
            if (kind not in ("search_query", "search_document") or not isinstance(texts, list)
                    or not 1 <= len(texts) <= MAX_BATCH
                    or any(not isinstance(t, str) or len(t) > MAX_CHARS for t in texts)):
                raise ValueError("invalid input")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return self.reply(400, {"error": "invalid input"})
        if not BUSY.acquire(blocking=False):
            return self.reply(503, {"error": "busy"})
        try:
            self.reply(200, {"embeddings": embed(texts, kind), "model": "multilingual-e5-small", "dimensions": 384})
        except Exception:
            logging.exception("inference failed")
            self.reply(500, {"error": "inference failed"})
        finally:
            BUSY.release()


def main():
    global TOKEN, SESSION, TOKENIZER
    import onnxruntime as ort
    from tokenizers import Tokenizer

    logging.basicConfig(level=logging.INFO)
    TOKEN = load_token(os.environ["EMBED_TOKEN_FILE"])
    ort.preload_dlls()  # CUDA libs from the venv nvidia-* wheels
    opts = ort.SessionOptions()
    SESSION = ort.InferenceSession(os.environ["EMBED_MODEL"], opts, providers=["CUDAExecutionProvider"])
    assert SESSION.get_providers()[0] == "CUDAExecutionProvider", SESSION.get_providers()  # never silent CPU
    TOKENIZER = Tokenizer.from_file(os.environ["EMBED_TOKENIZER"])
    server = ThreadingHTTPServer((os.environ["EMBED_HOST"], int(os.environ.get("EMBED_PORT", "18767"))), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(os.environ["EMBED_TLS_CERT"], os.environ["EMBED_TLS_KEY"])
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
