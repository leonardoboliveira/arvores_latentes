import os
import unittest

import numpy as np

from boilerplate import features
from boilerplate import loader as ldr
from boilerplate import mentions
from boilerplate.mentions_custom import increment_mention, increment_mention_pair

ROOT_PATH = "tests/"
TEST_FILE = ROOT_PATH + "cnn_0341.gold_conll"


class LoaderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass  # features.__GLOVE_FILE_NAME__ = "../" + features.__GLOVE_FILE_NAME__

    def test_append_mention_info(self):
        mention1 = mentions.Mention("0_0", ['x'], 1, 2)
        mention2 = mentions.Mention("0_0", ['x'], 3, 4)
        pair1 = mentions.MentionPair(mention1, mention2)

        mention3 = mentions.Mention("0_0", ['x'], 5, 6)
        mention4 = mentions.Mention("0_0", ['x'], 7, 8)
        pair2 = mentions.MentionPair(mention3, mention4)

        pairs = [pair1, pair2]
        vectors = [np.array([[5], [6], [7]]), np.array([[1], [2], [3]])]

        received = ldr._append_mention_info(pairs, vectors)
        expected = [[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 1, 2, 3]]

        self.assertListEqual(expected, received)

    def test_transform_conll_to_vectors(self):
        ldr.transform_conll_to_vectors(ROOT_PATH, ROOT_PATH, increment_mention, increment_mention_pair,
                                       features.make_vectors)
        self.assertTrue(os.path.isfile(TEST_FILE + "_in"))
        self.assertTrue(os.path.isfile(TEST_FILE + "_out"))

        os.unlink(TEST_FILE + "_in")
        os.unlink(TEST_FILE + "_out")

    def test_train_file_to_list(self):
        lines = ldr.train_file_to_list(TEST_FILE)
        self.assertEqual(356, len(lines))

    def test_get_document_name(self):
        lines = ["#begin document (nw/dev_09_c2e/00/dev_09_c2e_0000); part 000"]
        self.assertEqual("nw/dev_09_c2e/00/dev_09_c2e_0000", ldr.get_document_name(lines))


if __name__ == '__main__':
    unittest.main()
