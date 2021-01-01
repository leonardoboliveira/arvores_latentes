import unittest
from pre_training import predict_fine_tune as p
from pre_training import create_bert_ds
import numpy as np


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tokenizer = None

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = create_bert_ds.create_tokenizer()
        return self._tokenizer

    def test_example(self):
        with open("../../test_files/cnn_0341.gold_conll", "r") as f:
            lines = f.readlines()
        struct = p.build_struct(lines, self.get_tokenizer())
        example = struct.get_random_example()
        x1, x2, y = example
        self.assertEqual(1, len(x1))
        self.assertEqual(354, len(x1[0]))

    def test_example_bigger(self):
        lines = next(p.split_in_docs("../../test_files/cctv_0000.gold_conll"))
        struct = p.build_struct(lines, self.get_tokenizer())
        example = struct.get_random_example(True)
        x1, x2, y, first_pos = example
        self.assertEqual(2, len(x1))
        self.assertEqual(463, len(x1[0]))
        self.assertEqual(258, len(x1[1]))
        self.assertListEqual([1, 272], first_pos)

    def test_split_doc(self):
        docs = list(p.split_in_docs("../../test_files/cctv_0000.gold_conll"))
        self.assertEqual(12, len(docs))
        self.assertEqual(542, len(docs[0]))
        self.assertEqual(261, len(docs[1]))

    def test_merge_predictions_small(self):
        lines = next(p.split_in_docs("../../test_files/cnn_0341.gold_conll"))
        struct = p.build_struct(lines, self.get_tokenizer())
        predicted = [list(range(354))]
        merged = p.merge_predictions(predicted, struct)

        self.assertEqual(struct.size, len(merged))

    def test_merge_predictions_large(self):
        lines = next(p.split_in_docs("../../test_files/cctv_0000.gold_conll"))
        struct = p.build_struct(lines, self.get_tokenizer())
        predicted = [list(range(463)), list(range(258))]
        merged = p.merge_predictions(predicted, struct)

        self.assertEqual(struct.size, len(merged))
        self.assertEquals(0, np.sum(np.isnan(merged)))

    def test_clean_cluster(self):
        c = {51: [52],
             134: [139],
             137: [139],
             138: [139]}

        cleaned = p.clean_cluster(c)

        self.assertEqual(2, len(cleaned))
        self.assertIn(51, cleaned)
        self.assertIn(134, cleaned)


if __name__ == '__main__':
    unittest.main()
