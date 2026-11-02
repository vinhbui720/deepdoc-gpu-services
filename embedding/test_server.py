import unittest
from unittest.mock import patch

import server


class Contract(unittest.TestCase):
    def call(self, headers, auth_ok, body=b"{}"):
        class R:
            path = "/embed"
            def __init__(s):
                s.headers = headers
                s.rfile = __import__("io").BytesIO(body)
            def reply(s, status, data): s.result = status
        r = R()
        with patch.object(server.hmac, "compare_digest", return_value=auth_ok):
            server.Handler.do_POST(r)
        return r.result

    def test_ready_requires_auth_and_loaded_cuda(self):
        class Request:
            path = '/ready'
            headers = {'Authorization': 'Bearer token'}
            def reply(self, status, data): self.result = status
        request = Request()
        with patch.object(server.hmac, 'compare_digest', return_value=False):
            server.Handler.do_GET(request)
        self.assertEqual(request.result, 401)
        with patch.object(server.hmac, 'compare_digest', return_value=True):
            server.Handler.do_GET(request)
        self.assertEqual(request.result, 503)
        with patch.object(server, 'SESSION') as session, patch.object(server, 'TOKENIZER', object()), patch.object(server.hmac, 'compare_digest', return_value=True):
            session.get_providers.return_value = ['CUDAExecutionProvider']
            server.Handler.do_GET(request)
        self.assertEqual(request.result, 200)

    def test_401_413_400(self):
        self.assertEqual(self.call({"Content-Length": "9999999"}, False), 401)
        self.assertEqual(self.call({"Content-Length": "9999999"}, True), 413)
        self.assertEqual(self.call({"Content-Length": "2"}, True), 400)
        big = b'{"texts": [' + b",".join([b'"x"'] * 65) + b"]}"
        self.assertEqual(self.call({"Content-Length": str(len(big))}, True, big), 400)


if __name__ == "__main__":
    unittest.main()
