import unittest

from boilerplate import mentions as m

ROOT = "tests/"


class MentionsTestCase(unittest.TestCase):
    def setUp(self):
        with open("{}/cnn_0341.gold_conll".format(ROOT)) as f:
            self.lines = f.readlines()

    def test_get_index(self):
        m1 = m.Mention("0_7", ['a'], 1, 2)
        m2 = m.Mention("0_3", ['a'], 3, 4)
        m3 = m.Mention("0_5", ['a'], 5, 6)
        m4 = m.Mention("1_4", ['a'], 7, 8)

        m._get_index([m1, m2, m3, m4])

        self.assertEqual(1, m1.index)
        self.assertAlmostEqual(0.33, m1.mention_position, 1)
        self.assertEqual(2, m2.index)
        self.assertAlmostEqual(0.66, m2.mention_position, 1)

    def test_get_mention_pairs(self):
        pairs = m.get_mention_pairs(self.lines)
        self.assertEqual(339, len(pairs))

    def test_mention_sentence(self):
        self.assertEqual("Reporter :", m._mention_sentence(self.lines, 52))

    def test_build_mention_list(self):
        info = m.build_mention_list(self.lines)
        self.assertEqual(48, len(info))

    def test_get_mention(self):
        cluster_start, start_pos, cluster_end, end_pos = m._get_mention(self.lines)
        mention_cluster = m._create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos)

        self.assertEqual(48, len(mention_cluster))
        for mc in mention_cluster:
            self.assertGreaterEqual(mc[2], mc[1])

        self.assertListEqual(["0_9", 2, 11], mention_cluster[0])
        self.assertListEqual(["0_1", 4, 4], mention_cluster[1])
        self.assertListEqual(["0_4", 17, 21], mention_cluster[3])
        self.assertListEqual(["0_9", 20, 21], mention_cluster[4])

    def test_get_mention_words(self):
        self.assertListEqual("A former FBI informant accused of being a double agent".split(),
                             m.get_mention_words(self.lines, 2, 11))
        self.assertListEqual("the grand jury indictment".split(), m.get_mention_words(self.lines, 196, 199))
        print(" ".join(m.get_mention_words(self.lines, 121, 186)))

    def test_get_preceding_words(self):
        expected = ["the", "are", "Leung", "Katrina", "against"]
        self.assertListEqual(expected, m._get_preceding_words(self.lines, 24))

        expected = ["charges", "The", ".", "indicted", "been"]
        self.assertListEqual(expected, m._get_preceding_words(self.lines, 19))

        expected = ["former", "A"]
        self.assertListEqual(expected, m._get_preceding_words(self.lines, 4))

    def test_get_next_words(self):
        expected = ["in", "an", "alleged", "case", "of"]
        self.assertListEqual(expected, m._get_next_words(self.lines, 24))

        expected = ["been", "indicted", ".", "The", "charges"]
        self.assertListEqual(expected, m._get_next_words(self.lines, 12))

        expected = ["embarrassment", "."]
        self.assertListEqual(expected, m._get_next_words(self.lines, 352))


if __name__ == '__main__':
    unittest.main()
