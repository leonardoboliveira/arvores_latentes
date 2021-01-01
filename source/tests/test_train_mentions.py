import unittest
from training import train_mentions
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        mask = np.ones((10,))
        labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        reduced = train_mentions.reduce_mask(mask, labels, 0.5)

        self.assertGreaterEqual(reduced.sum(), 4)
        self.assertLessEqual(reduced.sum(), 6)

        reduced = train_mentions.reduce_mask(mask, labels, 0.99)
        self.assertEqual(1, reduced.sum())


if __name__ == '__main__':
    unittest.main()
