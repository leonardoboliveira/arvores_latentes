import unittest
from extension import arc_features


class MyTestCase(unittest.TestCase):
    def test_arc_index_diag(self):
        received = []

        for offset in range(1, arc_features.MAX_NUM_MENTIONS):
            for line in range(arc_features.MAX_NUM_MENTIONS - offset):
                received.append(arc_features.arc_index_diag(line, line + offset))

        expected = list(range(arc_features.MAX_NUM_PREDICTIONS))

        self.assertListEqual(expected, received)


if __name__ == '__main__':
    unittest.main()
