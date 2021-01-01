import unittest
from pre_training import create_bert_ds_cort


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tokenizer = None

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = create_bert_ds_cort.create_tokenizer()
        return self._tokenizer

    def test_group_lengths(self):
        lengts = [36, 17, 8, 27, 31, 8, 8, 24, 13, 8, 9, 14, 13, 47, 17, 7, 16, 14, 7, 8, 3, 8, 13, 51, 12, 25, 9, 9,
                  21]
        g = create_bert_ds_cort.Struture.calc_sentences_groups(lengts, 511)
        self.assertEqual(2, len(g))

    def test_parse(self):
        structure = create_bert_ds_cort.build_structure(self.get_tokenizer(), "../../test_files/cnn_0000.gold_conll")
        for s in structure.structure_map.values():
            for g in s.sentences_groups:
                lengths = [s.sentences_len_in_bert_tokens[idx] for idx in g]
                print(f"Group lengts:{lengths}")
                group_len = sum(lengths) + len(g) + 1
                self.assertGreaterEqual(512, group_len, f"Group too big:{g}")


if __name__ == '__main__':
    unittest.main()
