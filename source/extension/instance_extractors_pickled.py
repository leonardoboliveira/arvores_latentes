""" Extract instances and features from a corpus. """

import array
import multiprocessing
import sys
import os
import re

import mmh3
import numpy
from tqdm import tqdm
import compress_pickle as pickle
import tempfile
import concurrent.futures

__author__ = 'martscsn'

out_path = os.environ["PICKLED_PATH"]


def get_document_file(doc_id):
    name_re = re.compile(r".*\((.*)\); part (\d\d\d)")
    match = name_re.match(doc_id)
    pickle_file = os.path.join(out_path, f"{match.group(1)}_{match.group(2)}.lzma")

    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

    return pickle_file


class InstanceExtractor:
    """ Extract instances and their corresponding features from a corpus.

    Attributes:
        extract_substructures (function: CoNLLDocument ->
             list(list((Mention,Mention)))): Function for extracting the search
             space for a coreference resolution approach. The ith list in the
             nested list contains the search space for the ith substructure.
             The search space is represented as a nested list of mention pairs,
             which are candidate arcs in the graph to predict.
        mention_features (list(function: Mention -> str)): A list of features
            for mentions.
        pairwise_features (list(function: (Mention, Mention) -> str)): A list
            of features for mention pairs.
        cost_function (function: (Mention, Mention) -> int): A function
            assigning costs to mention pairs.
        labels (list(str)): A list of arc labels emplyoed by the approach.
            Defaults to the list containing only "+".
        convert_to_string_function (function): The function used to convert
            feature values to (unicode) strings. For Python 2 it is
            ``unicode``, for Python 3 it is ``string``.
    """

    def __init__(self,
                 extract_substructures,
                 mention_features,
                 pairwise_features,
                 cost_function,
                 labels=("+",)):
        """ Initialize instance and feature extraction.

        Args:
            extract_substructures (function: CoNLLDocument ->
                list(list((Mention,Mention)))): Function for extracting the
                search space for a coreference resolution approach. The ith
                list in the nested list contains the search space for the ith
                substructure. The search space is represented as a nested list
                of mention pairs, which are candidate arcs in the graph to
                predict.
            mention_features (list(function: Mention -> str)): A list of
                features for mentions.
            pairwise_features (list(function: (Mention, Mention) -> str)): A
                list of features for mention pairs.
            cost_function (function: (Mention, Mention) -> int): A function
                assigning costs to mention pairs.
            labels (list(str)): A list of arc labels emplyoed by the
                approach.
        """
        self.extract_substructures = extract_substructures
        self.mention_features = mention_features
        self.pairwise_features = pairwise_features
        self.cost_function = cost_function
        self.labels = labels
        self.id_to_doc_mapping = {}
        self.convert_to_string_function = str

    def extract(self, corpus):
        all_substructures = []
        arc_information = {}

        for doc in tqdm(corpus, desc="Loading results"):
            this_struct, this_info = self._load_info(doc)

            all_substructures += this_struct
            arc_information.update(this_info)

        return all_substructures, arc_information

    def _load_info(self, doc):
        file_name = get_document_file(doc.identifier)
        try:
            return pickle.load(file_name)
        except EOFError:
            print(f"Error loading file {file_name}")
            sys.exit(1)
