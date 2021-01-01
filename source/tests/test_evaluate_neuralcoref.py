import os
import unittest

import evaluate_neuralcoref as e

ROOT = "../"


class MyTestCase(unittest.TestCase):

    def test_get_mentions_spans(self):
        print(os.getcwd())
        stats = e.get_mentions_spans(
            os.path.join(ROOT, "../third_party/conll2012_boilerplate/tests/cnn_0341.gold_conll"))
        [mentions, spans] = stats

        self.assertTrue(len(mentions) > 0)
        self.assertTrue(len(spans) > 0)

        m = "A former FBI informant accused of being a double agent"
        self.assertIn(m, [m.mention for m in mentions])
        self.assertIn(m, [str(s) for s in spans[0]])


if __name__ == '__main__':
    unittest.main()
