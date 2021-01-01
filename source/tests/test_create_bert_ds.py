import unittest

from pre_training import create_bert_ds as ds


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with open("../../extra_files/p2.5_c2e_0049.gold_conll") as f:
            lines = f.readlines()
        mentions = ds.build_mention_list_cort(lines)
        self.assertEquals(64, len(list(mentions)))


if __name__ == '__main__':
    unittest.main()
