import unittest
from data_handling import create_mention_embedding_ts as cmet
import tempfile
import os
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_diag(self):
        labels = np.array([[0, 1, 5, 8, 10],
                           [0, 0, 2, 6, 9],
                           [0, 0, 0, 3, 7],
                           [0, 0, 0, 0, 4]])

        mask = np.ones((5, 5))
        l_labels, l_mask = cmet.linearized_diag(labels, mask, 5, 10)

        self.assertListEqual(list(range(1, 11)), list(l_labels))

    def test_something(self):
        os.environ["BERT_DB_PATH"] = r"D:\ProjetoFinal\data\devel\spacy"
        tmp = tempfile.mktemp()
        cmet.create_ts_file("../../test_files/cctv_0000.gold_conll", tmp)
        os.unlink(tmp)

    def test_to_one_example(self):
        embeddings = np.arange(150).reshape((5, 30))
        labels = np.triu(np.ones((5, 5)), 1)
        mask = np.triu(np.ones((5, 5)), 1)

        _, features = cmet.to_one_example(embeddings, labels, mask, True)
        out_mask = features["output_mask"]
        labels = features["labels"]

        self.assertListEqual(list(out_mask.int64_list.value), list(labels.int64_list.value))
        self.assertEqual(10, sum(out_mask.int64_list.value))
        self.assertListEqual([0, 1, 2, 3,
                              19, 20, 21,
                              37, 38,
                              54], list(np.where(out_mask.int64_list.value)[0]))


if __name__ == '__main__':
    unittest.main()
