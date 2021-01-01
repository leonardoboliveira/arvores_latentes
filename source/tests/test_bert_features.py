import unittest
import os
from extension import bert_features


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["BERT_DB_PATH"] = r"D:\ProjetoFinal\data\devel\sliding"

    def test_load_bert_db(self):
        e = bert_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 23, 24)
        self.assertAlmostEqual(-0.4378445, e[0], 10)

        e = bert_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 23, 25)
        self.assertAlmostEqual(-0.3735473, e[0], 10)

        e = bert_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 28, 29)
        self.assertAlmostEqual(-0.23204729, e[0], 9)


if __name__ == '__main__':
    unittest.main()
