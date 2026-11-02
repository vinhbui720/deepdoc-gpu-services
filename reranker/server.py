import hmac
import json
import logging
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore

HOST = os.environ['RERANK_HOST']
PORT = int(os.environ.get('RERANK_PORT', '18766'))
with open(os.environ['RERANK_TOKEN_FILE'], 'rb') as token_file:
    TOKEN = token_file.read().strip()
if len(TOKEN) < 32:
    raise RuntimeError('Token too short')
BUSY = BoundedSemaphore(1)
model = None
tokenizer = None
MAX_BODY = 262144

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logging.info('%s %s', self.address_string(), format % args)

    def reply(self, status, data):
        raw = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != '/ready':
            return self.reply(404, {'error': 'not found'})
        if not hmac.compare_digest(self.headers.get('Authorization', '').encode(), b'Bearer ' + TOKEN):
            return self.reply(401, {'error': 'unauthorized'})
        ready = model is not None and tokenizer is not None and next(model.parameters()).device.type == 'cuda'
        self.reply(200 if ready else 503, {'ready': ready})

    def do_POST(self):
        if self.path != '/rerank':
            return self.reply(404, {'error': 'not found'})
        auth = self.headers.get('Authorization', '')
        if not hmac.compare_digest(auth.encode(), b'Bearer ' + TOKEN):
            return self.reply(401, {'error': 'unauthorized'})
        length = self.headers.get('Content-Length', '')
        if not length.isdecimal() or not 0 < int(length) <= MAX_BODY:
            return self.reply(413, {'error': 'body limit'})
        try:
            data = json.loads(self.rfile.read(int(length)))
            query, passages = data['query'], data['passages']
            if not isinstance(query, str) or not 0 < len(query) <= 4096 or not isinstance(passages, list) or not 1 <= len(passages) <= 30 or any(not isinstance(p, str) or len(p) > 8192 for p in passages):
                raise ValueError('invalid input')
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return self.reply(400, {'error': 'invalid input'})
        if not BUSY.acquire(blocking=False):
            return self.reply(503, {'error': 'busy'})
        try:
            scores = []
            with torch.inference_mode():
                for i in range(0, len(passages), 10):
                    chunk = passages[i:i+10]
                    batch = tokenizer([query] * len(chunk), chunk, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
                    scores.extend(model(**batch).logits.float().view(-1).cpu().tolist())
            self.reply(200, {'scores': scores})
        except Exception:
            logging.exception('inference failed')
            self.reply(500, {'error': 'inference failed'})
        finally:
            BUSY.release()

if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    revision = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    repo = 'BAAI/bge-reranker-v2-m3'
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision, torch_dtype=torch.float16, local_files_only=True).to('cuda').eval()
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(os.environ["RERANK_TLS_CERT"], os.environ["RERANK_TLS_KEY"])
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()
