"""Isolated LAN HTTPS multilingual NLI endpoint; run with pinned local model."""
import hmac
import json
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.environ["NLI_MODEL_DIR"]
with open(os.environ["NLI_TOKEN_FILE"], "rb") as token_file:
    TOKEN = token_file.read().strip()
if len(TOKEN) < 32:
    raise RuntimeError("Token too short")
tokenizer = AutoTokenizer.from_pretrained(ROOT, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(ROOT, local_files_only=True).to("cuda").eval()
labels = {i: v.lower() for i, v in model.config.id2label.items()}
if set(labels.values()) != {"entailment", "neutral", "contradiction"}:
    raise RuntimeError("Unexpected model labels")
busy = BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log sensitive claims or bearer tokens.

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, {"status": "ready"} if self.path == "/health" else {})

    def do_POST(self):
        if self.path != "/nli":
            return self.reply(404, {})
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), b"Bearer " + TOKEN):
            return self.reply(401, {})
        length = self.headers.get("Content-Length", "")
        if not length.isdecimal() or not 0 < int(length) <= 65536:
            return self.reply(413, {})
        try:
            pairs = json.loads(self.rfile.read(int(length)))["pairs"]
            if not isinstance(pairs, list) or not 1 <= len(pairs) <= 20 or any(
                not isinstance(p, dict) or not isinstance(p.get("claim"), str) or
                not isinstance(p.get("premise"), str) or len(p["claim"]) > 2000 or
                len(p["premise"]) > 5000 or "hypothesis" in p for p in pairs
            ):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            return self.reply(400, {})
        if not busy.acquire(blocking=False):
            return self.reply(503, {})
        try:
            inputs = tokenizer([p["premise"] for p in pairs], [p.get("hypothesis", p["claim"]) for p in pairs],
                               padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = model(**inputs).logits
            self.reply(200, {"results": [{"label": labels[i], "confidence": round(float(prob[i]), 4)}
                                        for i, prob in zip(logits.argmax(-1).tolist(), logits.softmax(-1).tolist())]})
        except Exception:
            self.reply(500, {})
        finally:
            busy.release()


server = ThreadingHTTPServer((os.environ["NLI_HOST"], int(os.environ.get("NLI_PORT", "18766"))), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(os.environ["NLI_TLS_CERT"], os.environ["NLI_TLS_KEY"])
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
