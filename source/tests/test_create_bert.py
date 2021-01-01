import unittest

from pre_training import create_bert_ds as c
from boilerplate.mentions import Mention
import numpy as np


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tokenizer = c.create_tokenizer()

    def test_to_vectors(self):
        num_examples = 40
        X, Y, _ = c.to_vectors("../../test_files/cnn_0341.gold_conll", num_examples, self._tokenizer)

        self.assertEqual(40, len(X))
        self.assertEqual(354, len(X[0]))

    def test_structure(self):
        def mention(m_id, start, end):
            return Mention("0_" + str(m_id), ['w'], start, end)

        s = c.Structure(self._tokenizer, [". . . {}".format(x) for x in range(100)], 0)
        first_a = mention(0, 0, 3)
        first_b = mention(1, 3, 5)

        s.add_mention(first_a)
        s.add_mention(first_b)

        for i in range(4, 10):
            s.add_mention(mention(0, i * 2, i * 2))

        for i in range(4, 10):
            s.add_mention(mention(1, i * 2 + 1, i * 2 + 1))

        example_x1, example_x2, example_y = s.get_random_example()
        self.assertEquals(100, len(example_x1))
        self.assertEquals(100, len(example_y))

    def test_tree(self):
        t = c.Tree()
        a = t.add('A')
        b = t.add('B')

        aa = a.add_mention('AA')
        ab = a.add_mention('AB')

        aaa = aa.add_mention('AAA')

        self.assertEqual(2, len(t.root.children))
        self.assertSetEqual({'A', 'B'}, set([x.mention for x in t.root.children]))
        self.assertEqual(2, len(a.children))
        self.assertEqual(0, len(b.children))
        self.assertSetEqual({'AA', 'AB'}, set([x.mention for x in a.children]))
        self.assertEqual(0, len(ab.children))
        self.assertEqual(1, len(aa.children))
        self.assertEqual('AAA', aa.children[0].mention)

        if __name__ == '__main__':
            unittest.main()
