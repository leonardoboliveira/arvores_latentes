import array
import multiprocessing
import sys
import os

import mmh3
import numpy
from tqdm import tqdm
import compress_pickle as pickle
import tempfile
import re
from extension.arc_features_from_mentions import get_predictions
from training.train_mentions import get_model


def load_pickle(pickled):
    infos = []
    for saved, to_remove in tqdm(pickled, desc="Restoring from pickle"):
        # with open(saved, "rb") as f:
        try:
            infos.append(pickle.load(saved))
        except EOFError:
            print(f"Error loading file {saved}")
            exit(1)

        if to_remove:
            os.unlink(saved)
    return infos


def is_real_file(file_name):
    if not os.path.isfile(file_name):
        return False
    statinfo = os.stat(file_name)
    return statinfo.st_size > 0


def get_document_file(doc_id):
    if not "PICKLED_PATH" in os.environ:
        return None

    name_re = re.compile(r".*\((.*)\); part (\d\d\d)")
    match = name_re.match(doc_id)
    pickle_file = os.path.join(os.environ["PICKLED_PATH"], f"{match.group(1)}_{match.group(2)}.lzma")

    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

    return pickle_file


class InstanceExtractor():
    def __init__(self, model_name):
        self.predictor = None
        self.model_name = model_name

    def get_predictor(self):
        if self.predictor is None:
            self.predictor, _ = get_model()
            self.predictor.load_weights(self.model_name)

        return self.predictor

    def extract(self, corpus):
        """ Extract instances and features from a corpus.

        Args:
            corpus (Corpus): The corpus to extract instances and features from.

        Returns:
            A tuple which describes the extracted instances and their
            features. The individual components are:

            * substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            * arc_information (dict((Mention, Mention),
                                    ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
        """
        all_substructures = []
        arc_information = {}

        t_docs = tqdm(sorted(corpus.documents, key=lambda x: -len(x.system_mentions)), desc="Documents MT")

        with multiprocessing.Pool() as pool:
            for this_struct, this_info in pool.imap(self._extract_doc_info, t_docs):
                all_substructures.append(this_struct)
                arc_information.update(this_info)

        return all_substructures, arc_information

    def _extract_doc_info(self, document):
        predictions = get_predictions(document, self.get_predictor(), use_mask=True, use_cumsum=True)
        return list(predictions.keys()), predictions

    def _pickle_doc_info(self, document):
        remove_after = True
        pickle_file = get_document_file(document.identifier)
        if pickle_file:
            remove_after = False
            if is_real_file(pickle_file):
                return pickle_file, False
        else:
            _, pickle_file = tempfile.mkstemp()

        info = self._extract_doc_info(document)
        # with open(pickle_file, "wb") as f:
        pickle.dump(info, pickle_file, compression="lzma")

        return pickle_file, remove_after
