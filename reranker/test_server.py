import os
import tempfile
import unittest
from unittest.mock import patch

with tempfile.NamedTemporaryFile() as token:
    token.write(b'x' * 32)
    token.flush()
    os.environ['RERANK_HOST'] = '127.0.0.1'
    os.environ['RERANK_TOKEN_FILE'] = token.name
    import server


class Contract(unittest.TestCase):
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
        from unittest.mock import Mock
        gpu_model = Mock()
        gpu_model.parameters.return_value = iter([Mock(device=Mock(type='cuda'))])
        with patch.object(server, 'model', gpu_model), patch.object(server, 'tokenizer', object()), patch.object(server.hmac, 'compare_digest', return_value=True):
            server.Handler.do_GET(request)
        self.assertEqual(request.result, 200)

    def test_auth_and_body_limit(self):
        class Request:
            path = '/rerank'
            headers = {'Content-Length': '9999999'}
            def reply(self, status, data):
                self.result = status
        request = Request()
        with patch.object(server.hmac, 'compare_digest', return_value=False):
            server.Handler.do_POST(request)
        self.assertEqual(request.result, 401)
        with patch.object(server.hmac, 'compare_digest', return_value=True):
            server.Handler.do_POST(request)
        self.assertEqual(request.result, 413)


if __name__ == '__main__':
    unittest.main()
