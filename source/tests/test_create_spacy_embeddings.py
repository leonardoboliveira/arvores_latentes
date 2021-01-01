import unittest
from data_handling import create_spacy_embeddings
import tempfile
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Embedding:
    def __init__(self, word, idx, replace=False):
        self.text = word
        if replace:
            self.vector = idx
        else:
            self.vector = [0, 0, idx + 1]


class MyTestCase(unittest.TestCase):
    def test_real_case(self):
        work_file = tempfile.mktemp()

        create_spacy_embeddings.EMBEDDING_SIZE = 180

        def my_nlp(text):
            enc = OneHotEncoder(handle_unknown='ignore')
            X = np.array(text.split()).reshape(-1, 1)
            embeddins = enc.fit_transform(X).toarray()
            create_spacy_embeddings.EMBEDDING_SIZE = embeddins.shape[1]

            return [Embedding(X[i][0], embeddins[i], True) for i in range(embeddins.shape[0])]

        create_spacy_embeddings.nlp = my_nlp
        create_spacy_embeddings.create_embedding("../../test_files/p2.5_a2e_0040.gold_conll", work_file, 128, 32)
        os.unlink(work_file)

    def test_fake_doc_simple(self):
        doc_name = "my_doc"
        words = ["word1", "word2", "word3"]

        def my_nlp(text):
            return [Embedding(w, words.index(w)) for w in text.split()]

        create_spacy_embeddings.nlp = my_nlp
        create_spacy_embeddings.EMBEDDING_SIZE = 3
        doc = create_spacy_embeddings.Document(doc_name, 10, 1)

        for w in words:
            doc.add_line(". . . {}".format(w))

        self.assertEqual(len(words), len(doc.lines))
        doc.finish_doc()

        self.assertListEqual([0, 0, 1], doc.lines[0].chosen_embedding)

    def test_fake_doc_big_text(self):
        doc_name = "my_doc"
        words = ["word{}".format(i) for i in range(100)]

        def my_nlp(text):
            return [Embedding(w, words.index(w)) for w in text.split()]

        create_spacy_embeddings.nlp = my_nlp
        create_spacy_embeddings.EMBEDDING_SIZE = 3
        doc = create_spacy_embeddings.Document(doc_name, 10, 1)

        for w in words:
            doc.add_line(". . . {}".format(w))

        self.assertEqual(len(words), len(doc.lines))
        doc.finish_doc()

        self.assertListEqual([0, 0, 1], doc.lines[0].chosen_embedding)
        self.assertListEqual([0, 0, 51], doc.lines[50].chosen_embedding)
        self.assertListEqual([0, 0, 100], doc.lines[99].chosen_embedding)

    def test_fake_doc_sentences(self):
        doc_name = "my_doc"
        words = ["word_A_{}".format(i) for i in range(5)]
        words += ["word_B_{}".format(i) for i in range(5)]

        def my_nlp(text):
            return [Embedding(w, words.index(w)) for w in text.split()]

        create_spacy_embeddings.nlp = my_nlp
        create_spacy_embeddings.EMBEDDING_SIZE = 3
        doc = create_spacy_embeddings.Document(doc_name, 5, 1)

        for idx, w in enumerate(words):
            if idx == 4:
                doc.add_line("")
            doc.add_line(". . . {}".format(w))

        self.assertEqual(len(words) + 1, len(doc.lines))
        doc.finish_doc()

        self.assertListEqual([0, 0, 1], doc.lines[0].chosen_embedding)
        self.assertListEqual([0, 0, 4], doc.lines[3].chosen_embedding)
        self.assertIsNone(doc.lines[4].chosen_embedding)
        self.assertListEqual([0, 0, 5], doc.lines[5].chosen_embedding)
        self.assertListEqual([0, 0, 9], doc.lines[9].chosen_embedding)


if __name__ == '__main__':
    unittest.main()
