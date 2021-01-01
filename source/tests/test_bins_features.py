import unittest
import os
import tempfile
import shutil

from extension import bins_features


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["BERT_DB_PATH"] = r"D:\ProjetoFinal\data\devel\sliding"
        os.environ["BINS_FILE"] = r"D:\GDrive\Puc\Projeto Final\Code\extra_files\deciles.csv"

    def test_define_cumsum(self):
        self.assertListEqual([0, 0, 0, 0, 0], list(bins_features.define_cumsum(0)))
        self.assertListEqual([0, 0, 1, 1, 1], list(bins_features.define_cumsum(1)))
        self.assertListEqual([0, 1, 1, 2, 2], list(bins_features.define_cumsum(2)))
        self.assertListEqual([1, 1, 2, 2, 3], list(bins_features.define_cumsum(3)))
        self.assertListEqual([0, 1, 2, 3, 4], list(bins_features.define_cumsum(4)))
        self.assertListEqual([1, 2, 3, 4, 5], list(bins_features.define_cumsum(5)))
        self.assertListEqual([1, 2, 4, 5, 6], list(bins_features.define_cumsum(6)))
        self.assertListEqual([1, 3, 4, 6, 7], list(bins_features.define_cumsum(7)))

    def test_performance(self):
        for i in range(1000):
            self.test_load_bert_db()

    def test_load_bert_db(self):
        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 23, 24, return_embedding=True)
        self.assertAlmostEqual(-0.4378445, e[0], 10)

        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 23, 25, return_embedding=True)
        self.assertAlmostEqual(-0.3735473, e[0], 10)

        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 28, 29, return_embedding=True)
        self.assertAlmostEqual(-0.23204729, e[0], 9)

    def test_save_load_pickle(self):
        work_dir = tempfile.mkdtemp()
        os.environ["PICKLE_PATH"] = work_dir

        # Saving
        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 0, 1, return_embedding=True)
        self.assertAlmostEqual(-0.7252686, e[0], 10)

        # Saving
        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 001", 0, 1, return_embedding=True)
        self.assertAlmostEqual(-0.019680327, e[0], 10)

        bins_features._bert_db.clear()
        bins_features.bins = None

        # Loading
        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 000", 0, 1, return_embedding=True)
        self.assertAlmostEqual(-0.7252686, e[0], 10)

        # Loading
        x, e = bins_features.get_embedding("(bc/cctv/00/cctv_0000); part 001", 0, 1, return_embedding=True)
        self.assertAlmostEqual(-0.019680327, e[0], 10)

        shutil.rmtree(work_dir)


if __name__ == '__main__':
    unittest.main()
