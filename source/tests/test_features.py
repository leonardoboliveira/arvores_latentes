import unittest

import cort.coreference.features as features


class MyTestCase(unittest.TestCase):
    def test_features(self):
        features.BERT_DB_PATH = "../../extra_files/test_mention.encoded"
        vec = features.get_embedding("doc_name", 4, 5)
        self.assertListEqual([4.0], list(vec))

        vec = features.get_embedding("doc_name", 4, 6)
        self.assertListEqual([4.5], list(vec))


if __name__ == '__main__':
    unittest.main()
