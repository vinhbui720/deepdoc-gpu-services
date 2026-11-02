import unittest
from release import pins


class PinsTest(unittest.TestCase):
    def test_missing_revision_blocks(self):
        with self.assertRaises(ValueError):
            pins({name: {'revision': None, 'sha256': {}} for name in ('embedding', 'reranker', 'nli')})

    def test_missing_hash_blocks(self):
        with self.assertRaises(ValueError):
            pins({name: {'revision': 'a' * 40, 'sha256': {}} for name in ('embedding', 'reranker', 'nli')})
