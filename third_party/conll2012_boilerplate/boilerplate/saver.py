"""

This module is responsible for generating the output files for a given prediction. It will
use the original files to keep the format compatible
"""
import os


class LineInfo:
    """
    Class representing all operations (open/close brackets) for all clusters in a given line
    """

    def __init__(self):
        # Cluster id => Operation
        # Operations can be
        # 1 -> Open bracket
        # -1 -> Close bracket
        # 0 -> Open AND Close bracket
        self.operations = {}

    def add_operation(self, cluster_id, operation):
        """
        Adds the operation for that cluster_id in the current line
        :param cluster_id:
        :param operation:
        """
        if cluster_id not in self.operations:
            self.operations[cluster_id] = 0

        # By the definition, adding a close to a open leads to a open/close operation
        self.operations[cluster_id] += operation

    def __str__(self):
        """
        Method to translate this object into a string. In this implementation it will be translated
        into the CoNLL format for the mention coreference notation
        :return:
        """
        per_cluster = [self._to_simple_oper(k, v) for k, v in self.operations.items()]
        return "|".join(per_cluster)

    @staticmethod
    def _to_simple_oper(cluster_id, operation):
        """
        Translates the operation from number to string using open/close brackets.
        For example:
        '1', 1 => (1
        '2', -1 => 2)
        '3', 0 => (3)

        :param cluster_id:
        :param operation: integer representing the operation (1,-1,0)
        :return: string
        """
        output = str(cluster_id)
        if operation >= 0:
            output = "(" + output
        if operation <= 0:
            output += ")"

        return output


class Document:
    """
    Class representing a document and its mention clusters. This is the object that an algorithm should
    generate to allow the framework to save it in the proper CoNLL format.
    """

    def __init__(self, name, first_line_idx):
        self.name = name
        # Each cluster is a dictionary that maps start -> end for each mention in that cluster. Start/end are the
        # original document line number
        self.clusters = []

        # Line number => line info
        self.line_info = {}
        self.first_line_idx = first_line_idx
        self.next_line = 0
        self.next_line_generator = self._get_next_line()

    def add_cluster(self, cluster):
        """
        Adds a cluster to the document
        :param cluster: a dictionary that maps start_line -> [end_line] for each mention
        """
        self.clusters.append(cluster)
        cluster_id = len(self.clusters)

        # Adding one operation for key and another for value
        for key, values in cluster.items():
            self._add_operation(cluster_id, key, 1)
            for v in values:
                self._add_operation(cluster_id, v, -1)

    def _add_operation(self, cluster_id, line_number, operation):
        """
        Adds a single operation to the line
        :param cluster_id: generic cluster id
        :param line_number:
        :param operation: +1 -> open, -1 -> close
        :return:
        """
        if line_number not in self.line_info:
            self.line_info[line_number] = LineInfo()

        self.line_info[line_number].add_operation(cluster_id, operation)

    def _get_line_operations(self, line_number):
        """
        Gets the operation for a given line number
        :param line_number:
        :return:
        """
        if self.next_line < line_number:
            try:
                self.next_line = next(self.next_line_generator)
            except StopIteration:
                # If this exception occurs, it means that there are no more
                # lines to be generated. All following lines must be marked with -
                self.next_line += 1000

        if line_number == self.next_line:
            return str(self.line_info[line_number])

        return "-"

    def _get_next_line(self):
        """
        Generator for line numbers. Iterates over added lines information
        """
        for line_number in sorted(self.line_info):
            yield line_number

    def get_line(self, counter):
        return self.name + " " + self._get_line_operations(counter - self.first_line_idx)


def save_document(original_path, output_path, document):
    """
    Saves the document object to the output_path. It reads the original file to keep the common lines

    :param original_path:
    :param output_path:
    :param document: a Document object
    """
    with _open_original_file(original_path, document.name) as o:
        with _open_destination_file(output_path, document.name) as d:
            counter = 0
            for line in o:
                counter += 1

                if _write_as_is(line):  # Some lines are identical in both documents
                    d.write(line)
                else:  # Others are simply the name and the open/close mention (or a dash)
                    d.write(document.name + " " + document._get_line_operations(counter) + "\n")


def save_documents(in_file, out_file, docs):
    doc_iterator = iter(docs)
    with open(in_file, "r") as entrada, open(out_file, "w") as saida:
        counter = 0
        current_doc = next(doc_iterator)
        print(f"Writing doc {current_doc.name}")
        next_doc = next(doc_iterator)

        for line in entrada:
            counter += 1
            if _write_as_is(line):  # Some lines are identical in both documents
                write(saida, line)
            else:  # Others are simply the name and the open/close mention (or a dash)
                if (next_doc is not None) and (next_doc.first_line_idx <= counter):
                    current_doc = next_doc
                    print(f"Writing doc {current_doc.name}")
                    try:
                        next_doc = next(doc_iterator)
                        print("Moving to next document")
                    except StopIteration:
                        next_doc = None
                        print("No more docs")

                write(saida, current_doc.get_line(counter) + "\n")


def write(file, line):
    print(line)
    file.write(line)


def _open_original_file(original_path, name):
    """
    Opens (read) the gold_conll file associated with the given file
    :param original_path: root path (generally the 'annotation' folder)
    :param name: file name
    :return: a file object
    """
    return open("{}/{}.gold_conll".format(original_path, name), "r", encoding="utf8")


def _open_destination_file(output_path, name):
    """
    Opens (write) the file to be used as output. Will create any folder structe needed

    :param output_path:
    :param name: file
    :return: a file object
    """
    try:
        # This will create the folder structure to save the file
        os.makedirs(os.path.join(output_path, *name.split("/")[:-1]))
    except FileExistsError:
        pass  # Do nothing - folder structure already exists
    return open("{}/{}.output".format(output_path, name), "w")


def _write_as_is(line):
    """
    Checks if a line should be copied to the final document.
    The header, footer and blank lines are the only ones in this case

    :param line:
    :return: True if blank, header or footer. False otherwise
    """
    if line == "\n" or "#begin" in line or "#end" in line:
        return True
    return False
