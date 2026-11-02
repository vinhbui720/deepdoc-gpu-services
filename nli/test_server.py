import ast
import pathlib
import unittest


class Contract(unittest.TestCase):
    def test_source_parses_without_loading_gpu_model(self):
        source = pathlib.Path(__file__).with_name('server.py').read_text()
        ast.parse(source)
        self.assertIn('"hypothesis" in p', source)  # reject unbounded alternate hypothesis
        self.assertIn('self.path != "/ready"', source)
        self.assertIn('hmac.compare_digest(self.headers.get("Authorization", "").encode()', source)
        self.assertNotIn('self.path == "/health"', source)


if __name__ == '__main__':
    unittest.main()
