"""
This is an example of an algorithm tat reads the generated files and predicts somehow the outputs.
In this mock it will use exactly the supervised output as the prediction
"""
from boilerplate.saver import Document


def predict(file_name_x, file_name_y):
    """
    Builds a document based on the train files. This will not have any intelligence.
    It simply rebuild the original info

    :param file_name_x: file name of the X part of the information (features)
    :param file_name_y:  file name of the Y part of the information (supervised output)
    :return: a Document object
    """
    file_name, clusters = _create_mock(file_name_x, file_name_y)
    return _create_document(file_name, clusters)


def _create_mock(file_name_x, file_name_y):
    """
    Gets the information in the files and build a structure that contains all mention coreferences
    based on the file name y
    :param file_name_x:
    :param file_name_y:
    :return: file_name, dictionary of mention -> set of mentions. Each mention here is a tuple with (start, end)
    """
    corefs = {}

    with open(file_name_x, "r") as x_file:
        with open(file_name_y, "r") as y_file:
            original_file = next(x_file).strip("\n").strip("\r")

            for y_line in y_file:
                x_line = next(x_file)
                if y_line.strip("\n").strip("\r") == "0":
                    continue
                splitted = x_line.split(",")
                m1 = tuple([int(x) for x in splitted[0:2]])
                m2 = tuple([int(x) for x in splitted[2:4]])
                if m1 in corefs:
                    corefs[m1].add(m2)
                elif m2 in corefs:
                    corefs[m2].add(m1)
                else:
                    corefs[m1] = {m2}

    return original_file, corefs


def _create_document(file_name, mapping):
    """
    Creats a Document object based on the input

    :param file_name: document file name
    :param mapping: dictionary of tuples. Each entry must be mention head -> list of mentions.
            Each mention is a tuple of (start, end)
    :return: Document object
    """
    doc = Document(file_name)
    for key, values in mapping.items():
        cluster = {key[0]: key[1]}
        for v in values:
            if v[0] in cluster:
                print("This should not happen: same start in the same cluster,file_name:{}, start_line:{}".format(
                    file_name, v[0]))
            else:
                cluster[v[0]] = v[1]
        doc.add_cluster(cluster)

    return doc
