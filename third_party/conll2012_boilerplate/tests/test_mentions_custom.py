import unittest

import boilerplate.mentions_custom as mc


class MyTestCase(unittest.TestCase):
    def test_get_mention_length(self):
        received = mc._get_mention_length(["a", "b"])
        self.assertEqual("two", received)

    def test_mention_type(self):
        doc = mc.nlp("This is a common phrase that is being tested")
        mention = "common phrase"
        tp = mc._mention_type(doc, mention)
        self.assertListEqual([0, 0, 1, 0], list(tp))


if __name__ == '__main__':
    unittest.main()
