""" Extract instances and features from a corpus. """

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
import gc
import concurrent.futures

__author__ = 'martscsn'


def is_real_file(file_name):
    if (file_name is None) or (not os.path.isfile(file_name)):
        print("")
        return False
    statinfo = os.stat(file_name)
    return statinfo.st_size > 0


def get_document_file(doc_id):
    if "PICKLED_PATH" not in os.environ:
        return None

    name_re = re.compile(r".*\((.*)\); part (\d\d\d)")
    match = name_re.match(doc_id)
    pickle_file = os.path.join(os.environ["PICKLED_PATH"], f"{match.group(1)}_{match.group(2)}.lzma")

    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

    return pickle_file


def analyse_docs(docs):
    pickled = []
    missing = []
    for doc in tqdm(docs, desc="Checking docs"):
        pickle_file = get_document_file(doc.identifier)
        if is_real_file(pickle_file):
            pickled.append((pickle_file, False))
        else:
            missing.append(doc)

    return pickled, missing


# for python 2 multiprocessing
def unwrap_extract_doc(arg, **kwarg):
    return InstanceExtractor._extract_doc(*arg, **kwarg)


def split_features(in_fetures):
    o_features = []
    bert_features = []
    for name, feature in in_fetures:
        if "bert" in name:
            bert_features.append((name, feature))
        elif "bins" in name:
            for idx, val in enumerate(feature):
                o_features.append((f"{name}_{idx}", str(val)))
        elif "induced" in name:
            for val in feature:
                o_features.append((str(val), "1"))
        else:
            o_features.append((name, feature))
    return o_features, bert_features


offset = 768 * 00
mask = ((2 ** 24) - 1) - (offset)


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

        print(f"Mask:{bin(mask)}")
        if sys.version_info[0] == 2:
            self.convert_to_string_function = unicode
        else:
            self.convert_to_string_function = str

    def _pickle_doc_info(self, document):
        remove_after = True
        pickle_file = get_document_file(document.identifier)
        if pickle_file:
            remove_after = False
            if is_real_file(pickle_file):
                return pickle_file, False
        else:
            _, pickle_file = tempfile.mkstemp()

        # with open(pickle_file, "wb") as f:
        info = self._extract_doc_info(document)
        pickle.dump(info, pickle_file, compression="lzma")
        gc.collect()

        return pickle_file, remove_after

    def _extract_doc_info(self, document):
        result = self._extract_doc(document)
        return self._extract_arc_info(result)

    def _extract_arc_info(self, result):
        (doc_identifier,
         anaphors,
         antecedents,
         nonnumeric_features,
         numeric_features,
         numeric_vals,
         costs,
         consistency,
         nonnumeric_feature_mapping,
         numeric_feature_mapping,
         substructures_mapping) = result

        num_labels = len(self.labels)
        doc = self.id_to_doc_mapping[doc_identifier]
        arc_information = {}
        all_substructures = []

        for i in range(0, len(substructures_mapping) - 1):
            struct = []
            begin = substructures_mapping[i]
            end = substructures_mapping[i + 1]

            for pair_index in range(begin, end):
                if len(anaphors) <= pair_index or len(antecedents) <= pair_index:
                    print("Outside pair_index")

                if len(doc.system_mentions) <= anaphors[pair_index]:
                    print("Not found anaphor")
                    continue

                if len(doc.system_mentions) <= antecedents[pair_index]:
                    print("Not found antecedents")
                    continue

                arc = (doc.system_mentions[anaphors[pair_index]],
                       doc.system_mentions[antecedents[pair_index]])

                struct.append(arc)

                # find position of arc's features in document array
                nonnumeric_features_start = nonnumeric_feature_mapping[
                    pair_index]
                nonnumeric_features_end = nonnumeric_feature_mapping[
                    pair_index + 1]

                numeric_features_start = numeric_feature_mapping[pair_index]
                numeric_features_end = numeric_feature_mapping[
                    pair_index + 1]

                arc_information[arc] = \
                    ((nonnumeric_features[
                      nonnumeric_features_start:nonnumeric_features_end
                      ],
                      numeric_features[
                      numeric_features_start:numeric_features_end
                      ],
                      numeric_vals[
                      numeric_features_start:numeric_features_end
                      ]),
                     costs[num_labels * pair_index:num_labels * pair_index
                                                   + num_labels],
                     consistency[pair_index])

            all_substructures.append(struct)

        return (all_substructures, arc_information)

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

        self.id_to_doc_mapping.clear()
        for doc in corpus:
            self.id_to_doc_mapping[doc.identifier] = doc

        pickled = None
        infos = []
        if "EXTRACT_SINGLE_THREAD" in os.environ:
            infos = [self._extract_doc_info(d) for d in tqdm(corpus.documents, desc="Documents")]
        else:
            pickled, missing = analyse_docs(corpus.documents)
            # t_docs = tqdm(sorted(missing, key=lambda x: -(len(x.system_mentions) ** 2)), desc="Documents MT")
            t_docs = tqdm(missing, desc="Documents MT")

            if len(missing) > 0:
                with multiprocessing.Pool(maxtasksperchild=1) as pool:
                    pickled += pool.map(self._pickle_doc_info, t_docs)

            print(f"Restoring from {os.environ['PICKLED_PATH']}")
            for saved, to_remove in tqdm(pickled, desc="Restoring from pickle"):
                # with open(saved, "rb") as f:
                try:
                    infos.append(pickle.load(saved))
                except EOFError:
                    print(f"Error loading file {saved}")
                    exit(1)
                if to_remove:
                    os.unlink(saved)

        for this_struct, this_info in tqdm(infos, desc="Merging results"):
            all_substructures += this_struct
            arc_information.update(this_info)

        # in python 2, array.array does not support the buffer interface
        if sys.version_info[0] == 2:
            for arc in arc_information:
                feats, cost, cons = arc_information[arc]
                arc_information[arc] = (
                    (numpy.array(feats[0], dtype=numpy.uint32),
                     numpy.array(feats[1], dtype=numpy.uint32),
                     numpy.array(feats[2], dtype="float32")),
                    numpy.array(cost, dtype=float),
                    cons)

        return all_substructures, arc_information

    def _extract_doc(self, doc):
        cache = {}
        substructures = self.extract_substructures(doc)

        mentions_to_ids = {}

        for i, mention in enumerate(doc.system_mentions):
            mentions_to_ids[mention] = i

        anaphors = array.array('H')
        antecedents = array.array('H')
        costs = array.array('H')
        consistency = array.array('B')
        nonnumeric_feature_mapping = array.array('I')
        numeric_feature_mapping = array.array('I')
        substructures_mapping = array.array('I')
        nonnumeric_features = array.array('I')
        numeric_features = array.array('I')
        numeric_vals = array.array("f")

        nonnumeric_feature_mapping.append(0)
        numeric_feature_mapping.append(0)
        substructures_mapping.append(0)

        for struct in substructures:
            # skip empty
            if not struct:
                continue

            for arc in tqdm(struct, desc="Arc"):
                # ids for anaphor and antecedent
                anaphors.append(mentions_to_ids[arc[0]])
                antecedents.append(mentions_to_ids[arc[1]])

                # cost for each label
                for label in self.labels:
                    costs.append(self.cost_function(arc, label))

                # is decision to make them coreferent consistent with gold?
                consistency.append(arc[0].decision_is_consistent(arc[1]))

                # features for the arc: stored in array which applies to whole
                # document
                (arc_nonnumeric_features, arc_numeric_features,
                 arc_numeric_vals) = self._extract_features(arc, cache)

                nonnumeric_features.extend(arc_nonnumeric_features)
                numeric_features.extend(arc_numeric_features)
                numeric_vals.extend(arc_numeric_vals)

                # auxiliary arrays that store the position of features for arcs
                # in the document array
                nonnumeric_feature_mapping.append(
                    nonnumeric_feature_mapping[-1] + len(
                        arc_nonnumeric_features))
                numeric_feature_mapping.append(
                    numeric_feature_mapping[-1] + len(arc_numeric_features))

            # store position of substructures in document array
            substructures_mapping.append(substructures_mapping[-1] +
                                         len(struct))

        return (doc.identifier,
                anaphors,
                antecedents,
                nonnumeric_features,
                numeric_features,
                numeric_vals,
                costs,
                consistency,
                nonnumeric_feature_mapping,
                numeric_feature_mapping,
                substructures_mapping)

    def _extract_features(self, arc, cache):
        anaphor, antecedent = arc
        inst_feats = []
        bert_features = []
        numeric_types = {"float", "int", "list", "Series"}

        cojoin = True
        efi = False
        if "SKIP_FEATURE_INDUCTION" in os.environ:
            cojoin = os.environ["SKIP_FEATURE_INDUCTION"] == "COJOIN"
            efi = os.environ["SKIP_FEATURE_INDUCTION"] == "EFI"

        if not antecedent.is_dummy():
            # mention features
            for mention in [anaphor, antecedent]:
                if mention not in cache:
                    cache[mention] = [feature(mention) for feature
                                      in self.mention_features]

            ana_features, bert_ana_features = split_features(cache[anaphor])
            ante_features, bert_ante_features = split_features(cache[antecedent])

            # first: non-numeric features (categorial, boolean)
            inst_feats += ["ana_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ana_features if type(val).__name__ not in
                           numeric_types]

            len_ana_features = len(inst_feats)

            inst_feats += ["ante_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ante_features if type(val).__name__ not in
                           numeric_types]

            if cojoin:
                # print("Doing EFI")
                # concatenated features
                inst_feats += ["ana_" + ana_info[0] + "=" +
                               self.convert_to_string_function(ana_info[1]) +
                               "^ante_" + ante_info[0] + "=" +
                               self.convert_to_string_function(ante_info[1])
                               for ana_info, ante_info in
                               zip(ana_features, ante_features)]

            # pairwise features
            pairwise_features, bert_pairwise_features = split_features([feature(anaphor, antecedent) for feature
                                                                        in self.pairwise_features])
            inst_feats += [feature + "=" +
                           self.convert_to_string_function(val) for feature, val
                           in pairwise_features
                           if val and type(val).__name__ not in numeric_types]

            for k, v in bert_pairwise_features:
                if "bert" in k:
                    bert_features.append(v)

            for features in [bert_ana_features, bert_ante_features]:
                for f in features:
                    if "bert" in f[0]:
                        bert_features.append(f[1])

            # feature combinations
            fine_type_indices = {len_ana_features * i for i
                                 in [0, 1, 2]}

            if cojoin:
                inst_feats += [
                    inst_feats[i] + "^" + word for i in fine_type_indices
                    for j, word in enumerate(inst_feats)
                    if j not in fine_type_indices
                ]
        else:
            for mention in [anaphor]:
                if mention not in cache:
                    cache[mention] = [feature(mention) for feature
                                      in self.mention_features]

            _, bert_ana_features = split_features(cache[anaphor])
            for f in bert_ana_features:
                if "bert" in f[0]:
                    bert_features.append(f[1])

        # print(f"Shapes:{[x.shape for x in bert_features]}")
        vals = [] if len(bert_features) == 0 else numpy.concatenate(bert_features)
        all_numeric_feats = array.array('I', list(range(len(vals))))
        numeric_vals = array.array("f", vals)

        def hash_word(word):
            hash = mmh3.hash(word.encode("utf-8"))
            hash = (hash % mask) + offset
            return hash

        if efi:
            from extension.induced_features import induce_features
            inst_feats = induce_features(inst_feats)

        all_nonnumeric_feats = array.array('I', [hash_word(w) for w in inst_feats])

        return all_nonnumeric_feats, all_numeric_feats, numeric_vals
