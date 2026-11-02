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
