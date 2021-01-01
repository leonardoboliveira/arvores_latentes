import unittest

from analysis import viewer


class MyTestCase(unittest.TestCase):

    def test_create_arc(self):
        iterator = viewer.read_arcs("../../test_files/cctv_0000.ante")
        arc = next(iterator)

        self.assertTupleEqual((32, 37), arc[0])
        self.assertTupleEqual((28, 28), arc[1])

        for _ in iterator:
            pass

    def test_create_clusters(self):
        arcs = [
            [(3, 4), (1, 2)],
            [(5, 6), (3, 4)],
            [(10, 11), (8, 9)],
            [(12, 13), (1, 2)],
        ]
        clusters = viewer.create_clusters(arcs)

        expected = {2, 4}
        received = set([len(x) for x in clusters])

        self.assertSetEqual(expected, received)

    def test_mark_text(self):
        text = "Just a text"
        clusters = [[(1, 1)]]

        expected = "Just <cluster_0>a</cluster_0> text"

        self.assertEqual(expected, viewer.mark_text(text, clusters))

        text = "Today , let 's turn our attention " \
               "to a road cave - in accident " \
               "that happeded in Beijing over " \
               "the hooliday ."
        clusters = [
            [(0, 0)],
            [(8, 20)],
            [(17, 17)],
            [(19, 20)]
        ]
        expected = "<cluster_0>Today</cluster_0> , let 's turn our attention " \
                   "to <cluster_1>a road cave - in accident " \
                   "that happeded in <cluster_2>Beijing</cluster_2> over " \
                   "<cluster_3>the hooliday</cluster_3></cluster_1> ."

        self.assertEqual(expected, viewer.mark_text(text, clusters))

        clusters = [
            [(0, 0)],
            [(19, 20)],
            [(17, 17)],
            [(8, 20)]
        ]
        expected = "<cluster_0>Today</cluster_0> , let 's turn our attention " \
                   "to <cluster_3>a road cave - in accident " \
                   "that happeded in <cluster_2>Beijing</cluster_2> over " \
                   "<cluster_1>the hooliday</cluster_1></cluster_3> ."

        self.assertEqual(expected, viewer.mark_text(text, clusters))


if __name__ == '__main__':
    unittest.main()
