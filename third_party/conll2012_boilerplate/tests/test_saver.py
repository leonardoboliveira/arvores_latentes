import os
import unittest

from boilerplate import saver as s

ROOT = "../tests/"


class MyTestCase(unittest.TestCase):
    def test_lineinfo(self):
        info = s.LineInfo()
        info.add_operation(1, 1)
        info.add_operation(2, 1)
        info.add_operation(1, -1)
        info.add_operation(3, -1)

        self.assertEqual("(1)|(2|3)", str(info))

    def test_document(self):
        doc = s.Document("doc_name", 0)
        doc.add_cluster({1: [2], 4: [4], 5: [10]})
        doc.add_cluster({4: [7]})

        expected = ["(1", "1)", "-", "(1)|(2", "(1", "-", "2)", "-", "-", "1)"]
        received = [doc._get_line_operations(i + 1) for i in range(len(expected))]

        self.assertListEqual(expected, received)

    def test_save_document(self):
        doc = s.Document("cnn_0341", 0)
        doc.add_cluster({1: [2], 4: [4], 5: [10]})
        doc.add_cluster({4: [7]})
        s.save_document("{}/".format(ROOT), "./", doc)
        self.assertTrue(os.path.isfile("./cnn_0341.output"))
        expected = ["#begin document (bn/cnn/03/cnn_0341); part 000", "cnn_0341 -", "cnn_0341 -", "cnn_0341 (1)|(2",
                    "cnn_0341 (1", "cnn_0341 -", "cnn_0341 2)", "cnn_0341 -", "cnn_0341 -", "cnn_0341 1)", ]

        with open("./cnn_0341.output", "r") as f:
            lines = f.readlines()
            for i in range(len(expected)):
                self.assertEqual(lines[i].strip("\n"), expected[i])

        os.unlink("./cnn_0341.output")

    if __name__ == '__main__':
        unittest.main()
