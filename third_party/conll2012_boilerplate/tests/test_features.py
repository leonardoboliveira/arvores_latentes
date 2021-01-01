import unittest

import numpy as np

from boilerplate import features as f
from boilerplate.loader import train_file_to_list

ROOT_PATH = "tests/"
TEST_FILE = ROOT_PATH + "cnn_0341.gold_conll"


class FeaturesTestCase(unittest.TestCase):
    def test_get_average_vector(self):
        words = "This is doc one. Line two doc one. ".split()

        # Building a very simple mock model for this phrase. One hot for the position in the sentence
        model = {}
        for i in range(len(words)):
            w = words[i].lower().replace(".", "")  # Punctuation show not exists
            if w in model:
                v = model[w]
            else:
                v = np.zeros((50, 1))
            v[i] += 1
            model[w] = v

        features = f.FeatureMapper(mapper(model), [])
        # Testing behavior for empty list
        features._get_average_vector([])
        # Real testing here
        v = features._get_average_vector(words)

        # words 'doc' and 'one' are repeated
        base = 1 / 8
        expected = ([base] * 8)
        for i in [2, 3, 6, 7]:  # This is accounting for the repetition
            expected[i] *= 2

        for i in range(len(expected)):
            self.assertEqual(expected[i], v[i])

        for i in range(i + 1, 50):
            self.assertEqual(0, v[i])

    def test_calculate_docs_average(self):
        features = f.FeatureMapper(mapper({}), train_file_to_list(TEST_FILE))

        docs = features._calculate_docs_average()
        self.assertEqual(1, len(docs))

    def test_get_vector(self):
        features = f.FeatureMapper(mapper({"teste": np.ones((50, 1))}), [])

        v = features._get_vector("Teste.")
        self.assertEqual(50, (np.ones((50, 1)) == v).sum())

        v = features._get_vector("NOT_EXISTING")
        self.assertEqual(50, (np.zeros((50, 1)) == v).sum())

    def test_get_documents(self):
        docs = f._get_documents(train_file_to_list(TEST_FILE))
        self.assertEqual(1, len(docs))
        self.assertEqual("A former FBI informant accused of being a double agent has been indicted. ", docs[0][0])

    def test_document_dictionary(self):
        docs = f._document_dictionary(train_file_to_list(TEST_FILE))
        self.assertEqual(1, len(docs))
        self.assertEqual(1710, len(docs[0]))

    def test_merge_document_text(self):
        docs = [["This is doc one. ", "Line two doc one. "], ["This is other doc. ", "Second Line "]]

        expected = {0: "This is doc one. Line two doc one. ", 1: "This is other doc. Second Line "}

        self.assertDictEqual(expected, f._merge_document_text(docs))


def mapper(word_mapping):
    return lambda x: word_mapping[x]


if __name__ == '__main__':
    unittest.main()
