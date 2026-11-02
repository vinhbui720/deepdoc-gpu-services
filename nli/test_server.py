import ast
import pathlib
import unittest


class Contract(unittest.TestCase):
    def test_source_parses_without_loading_gpu_model(self):
        source = pathlib.Path(__file__).with_name('server.py').read_text()
        ast.parse(source)
        self.assertIn('"hypothesis" in p', source)  # reject unbounded alternate hypothesis


if __name__ == '__main__':
    unittest.main()
