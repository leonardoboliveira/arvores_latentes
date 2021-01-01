import unittest

from pre_training import create_pretraining_data as e
from bert import tokenization
from boilerplate.mentions import Mention


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tokenizer = c.create_tokenizer_from()

    def test_find_occurrence(self):
        start, end = e.find_occurrence([1, 2, 3, 4, 5, 6, 7], [3, 4, 5])
        self.assertEqual(2, start)
        self.assertEqual(4, end)

    def test_mention_to_instance(self):
        tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt")
        mention = Mention("0_1", ["a", "bunch", "of", "words"], 2, 5)
        mention.mention_sentence = "Most of this sentence is a bunch of words, but it has a purpose"

        instance = e.mention_to_instance(tokenizer, mention)
        self.assertEqual(13, len(instance.tokens))
        self.assertEquals(1, len(instance.masked_lm_positions))
        self.assertEquals(1, instance.masked_lm_labels[0])
        self.assertEquals(5, instance.masked_lm_positions[0])


if __name__ == '__main__':
    unittest.main()
