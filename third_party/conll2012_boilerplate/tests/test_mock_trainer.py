import os
import unittest

from boilerplate import loader as ldr
from boilerplate import mock_trainer
from boilerplate.features import make_vectors
from boilerplate.mentions_custom import increment_mention, increment_mention_pair

ROOT = "tests/"


class MockTrainerTestCase(unittest.TestCase):
    def test_create_mock(self):
        ldr.transform_conll_to_vectors(ROOT, ROOT, increment_mention, increment_mention_pair, make_vectors)

        file_name, mock = mock_trainer._create_mock("{}/cnn_0341.gold_conll_in".format(ROOT),
                                                    "{}/cnn_0341.gold_conll_out".format(ROOT))

        self.assertEqual("bn/cnn/03/cnn_0341", file_name)
        self.assertEqual(35, len(mock))
        expected = (20, 21)
        expected2 = (2, 11)
        self.assertTrue(expected in mock)
        self.assertTrue(expected2 in mock[expected])

        os.unlink("{}/cnn_0341.gold_conll_in".format(ROOT))
        os.unlink("{}/cnn_0341.gold_conll_out".format(ROOT))

    def test_create_document(self):
        mapping = {(1, 2): [[(3, 4), (5, 6)]], (7, 8): [[(9, 10), (11, 12)]]}
        doc = mock_trainer._create_document("X", mapping)

        self.assertEqual("X", doc.name)
        self.assertEqual(2, len(doc.clusters))


if __name__ == '__main__':
    unittest.main()
