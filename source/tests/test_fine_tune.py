import unittest
from pre_training import fine_tune_bert as tune
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_something(self):
        df = pd.DataFrame([[-1, 0, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0]])
        df = tune.fix_y_df(df)
        self.assertEqual(3, len(df))
        self.assertEqual(0, df.min().min())
        self.assertEqual(4, df.max().max())

        # -1 and 0 and swapped
        # increase index so there are no negative
        self.assertListEqual([1, 0, 2, 3, 4], list(df.iloc[0, :].values))


if __name__ == '__main__':
    unittest.main()
