"""
This module will parse the files in the CoNLL format.
The main function is trainfile_to_vectors

"""

import os
import re

from tqdm import tqdm

from boilerplate import mentions


def trainfile_to_vectors(path, increment_mention, increment_mention_pair, make_vectors):
    """
    Given one file, returns the input and output vectors to be passed to a learning algo
    :param path: file path to be used
    :param increment_mention: method to add information to the mention
    :param increment_mention_pair: method to add information to the mention pair
    :param make_vectors: method to build the vectors
    :return: [input_vector, output_vector, document_name]
    """
    train_list = train_file_to_list(path)
    pairs = mentions.get_mention_pairs(train_list, increment_mention, increment_mention_pair)
    input_vector, output_vector = make_vectors(pairs, train_list=train_list)
    input_vector = _append_mention_info(pairs, input_vector)
    return input_vector, output_vector, get_document_name(train_list)


def process_dir(path_in, path_out, callback):
    with tqdm(os.walk(path_in), desc="folders")as pb:
        for r, d, f in pb:
            pb.set_description("folder:...{}".format(r[-20:]))
            for file_name in tqdm(f, desc="files"):
                if not file_name.endswith("_conll"):
                    continue
                v_in, v_out, doc_name = callback(os.path.join(r, file_name))

                if len(v_in) > 0 and len(v_out) > 0:
                    _save_to_file(v_in, path_out, file_name + "_in", doc_name)
                    _save_to_file(v_out, path_out, file_name + "_out")


def transform_conll_to_vectors(path_in, path_out, increment_mention, increment_mention_pair, make_vectors):
    """
    Walks the input path looking for *_conll files. If any file is found, it is processed and two files are generated
    into the path_out root.
    A file with pattern [original_name]_in with the input vectors and [original_name]_out with the output vectors
    :param path_in: root folder to be searched. Files can be in multiple sub-folders
    :param path_out: output folder
    :param increment_mention: method to add information to the mention
    :param increment_mention_pair: method to add information to the mention pair
    :param make_vectors: method to build the vectors
    """

    process_dir(path_in, path_out,
                lambda x: trainfile_to_vectors(x, increment_mention, increment_mention_pair, make_vectors))


def train_file_to_list(file):
    """
    :param file: file name to be read
    :return: list of all lines
    """
    with open(file, "r", encoding="utf8") as f:
        return f.readlines()


def _save_to_file(vector, path, file_name, doc_name=None, as_int=False, as_hex=True):
    """
    Saves a vector into a file if the vector is not empty
    :param vector: list of lists of values. Can be numpy arrays
    :param path: folder to save
    :param file_name: file name to use
    """
    if len(vector) == 0:  # Do not create empty files
        return
    with open(os.path.join(path, file_name), "w") as f:
        if doc_name:
            f.write(doc_name + "\n")

        format = "{}"
        if as_int:
            format = "{:.0f}"
        elif as_hex:
            format = "0x{:x}"

        for line in vector:
            f.write(",".join([format.format(i).replace("0x-1", "-0x1") for i in line]) + "\n")


def _append_mention_info(pairs, input_vectors):
    """
    Append pair information into the vector to be saved in the disk
    :param pairs: list of mention pairs
    :param input_vectors: list of vectors
    :return: a list of all information appended
    """
    appended = []
    for i in range(len(pairs)):
        vec_as_list = list([x[0] for x in input_vectors[i]])
        if len(vec_as_list) == 0:
            continue
        appended.append(list(pairs[i].get_info_vector()) + vec_as_list)

    return appended


def get_document_name(train_list):
    """
    The the document name from the first line.
    :param train_list: list of all lines in the document
    :return: the document name
    """
    name_re = re.compile(r".*\((.*)\).*")
    match = name_re.match(train_list[0])
    return match.group(1)
