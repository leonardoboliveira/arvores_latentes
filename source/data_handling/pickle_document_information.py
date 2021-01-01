import array
import multiprocessing
import os
import re
import sys
import glob
import gc

import numpy
from tqdm import tqdm
import compress_pickle as pickle
import logging
import tempfile
import mmh3

from cort.core import corpora
from cort.util import import_helper
from cort.coreference.cost_functions import cost_based_on_consistency
from cort.core import mention_extractor

from extension import antecedent_trees

mask = ((2 ** 24) - 1)
cores = int(os.cpu_count() / 4)


def split_docs(docs):
    import heapq

    class MyStruct:
        def __init__(self):
            self.docs = []
            self.total = 0

        def add_doc(self, doc):
            self.docs.append(doc)
            self.total += len(doc.system_mentions) ** 2

        def __lt__(self, other):
            return self.total < other.total

    mentions_counts = [len(x.system_mentions) for x in docs]
    threshold = numpy.mean(mentions_counts) + 4 * numpy.std(mentions_counts)
    logging.info(f"Mentions Threshold: {threshold}")

    li = [MyStruct() for i in range(cores)]
    outliers = []

    for doc in sorted(docs, key=lambda x: -len(x.system_mentions)):
        if len(doc.system_mentions) > threshold:
            outliers.append(doc)
            continue

        min_list = heapq.heappop(li)
        min_list.add_doc(doc)
        heapq.heappush(li, min_list)

    li = [x for x in li if x.total > 0]
    logging.info(f"Doc sizes:{[x.total for x in li]}")
    logging.info(f"Outliers sizes:{[len(x.system_mentions) for x in outliers]}")

    normal = [x.docs for x in li]

    if len(normal) < cores / 2:
        outliers += [item for sublist in normal for item in sublist]
        normal = []

    print(f"Normal len:{len(normal)}, Outlier len:{len(outliers)}")

    return normal, outliers


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
                 out_path,
                 skip_if_exists,
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
        self.out_path = out_path
        self.skip_if_exists = skip_if_exists

    def assert_folder_exists(self, doc_id):
        name_re = re.compile(r".*\((.*)\); part (\d\d\d)")
        match = name_re.match(doc_id)
        pickle_file = os.path.join(self.out_path, f"{match.group(1)}_{match.group(2)}")

        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)

        return pickle_file

    def _pickle_block_info(self, block):
        files = []
        for doc in tqdm(block, desc="Doc Block", mininterval=1):
            files.append(self._pickle_doc_info(doc))
        logging.debug("Finished Block")
        return files

    def _pickle_doc_info(self, document, pool=None):
        file_name = self.assert_folder_exists(document.identifier)
        if self.skip_if_exists and len(glob.glob(file_name + ".*")) > 0:
            print(f"File already exists. Skipping {file_name}")
            return file_name
        info = self._extract_doc_info(document, pool)
        pickle.dump(info, file_name, compression="lzma")
        gc.collect()
        return file_name

    def _extract_doc_info(self, document, pool):
        result = self._extract_doc(document, pool)
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
                    print("Not echo $Bfound anaphor")
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
        self.id_to_doc_mapping.clear()

        docs = []
        for doc in tqdm(corpus, desc="Creating folders"):
            self.id_to_doc_mapping[doc.identifier] = doc
            file_name = self.assert_folder_exists(doc.identifier)
            if self.skip_if_exists and len(glob.glob(file_name + ".*")) > 0:
                continue
            docs.append(doc)

        if len(docs) == 0:
            print("All docs already exists")
            return

        with multiprocessing.Pool() as pool:
            bulk, outliers = split_docs(docs)
            pool.map(self._pickle_block_info, bulk)
            print("Now processing outliers")
            for x in outliers:
                self._pickle_doc_info(x, pool)

    def _extract_doc(self, doc, pool):
        cache = {}
        substructures = self.extract_substructures(doc)

        mentions_to_ids = {}

        for i, mention in enumerate(doc.system_mentions):
            mentions_to_ids[mention] = i

        anaphors = array.array('H')
        antecedents = array.array('H')
        costs = array.array('H')
        consistency = array.array('B')
        nonnumeric_feature_mapping = array.array('L')
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

            extracted_features = self._get_features(struct, pool)

            for idx, arc in enumerate(struct):
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
                 arc_numeric_vals) = extracted_features[idx]

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
        skip_features_induction = "SKIP_FEATURE_INDUCTION" in os.environ

        def split_features(in_fetures):
            o_features = []
            bert_features = []
            for name, feature in in_fetures:
                if "bert" in name:
                    bert_features.append((name, feature))
                elif "bins" in name:
                    for idx, val in enumerate(feature):
                        o_features.append((f"{name}_{idx}", str(val)))
                else:
                    o_features.append((name, feature))
            return o_features, bert_features

        if (not antecedent.is_dummy()) and (not anaphor.is_dummy()):
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

            if not skip_features_induction:
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

            if not skip_features_induction:
                inst_feats += [
                    inst_feats[i] + "^" + word for i in fine_type_indices
                    for j, word in enumerate(inst_feats)
                    if j not in fine_type_indices
                ]
        elif not anaphor.is_dummy():
            for mention in [anaphor]:
                if mention not in cache:
                    cache[mention] = [feature(mention) for feature
                                      in self.mention_features]

            _, bert_ana_features = split_features(cache[anaphor])
            for f in bert_ana_features:
                if "bert" in f[0]:
                    bert_features.append(f[1])
        elif not antecedent.is_dummy():
            for mention in [antecedent]:
                if mention not in cache:
                    cache[mention] = [feature(mention) for feature
                                      in self.mention_features]

            _, bert_ana_features = split_features(cache[antecedent])
            for f in bert_ana_features:
                if "bert" in f[0]:
                    bert_features.append(f[1])

        # print(f"Shapes:{[x.shape for x in bert_features]}")
        vals = [] if len(bert_features) == 0 else numpy.concatenate(bert_features)
        all_numeric_feats = array.array('I', list(range(len(vals))))
        numeric_vals = array.array("f", vals)

        def hash_word(word):
            hash = mmh3.hash(word.encode("utf-8"))
            hash = (hash % mask)
            return hash

        all_nonnumeric_feats = array.array('I', [hash_word(w) for w in inst_feats])

        return all_nonnumeric_feats, all_numeric_feats, numeric_vals

    def _get_features(self, struct, pool):
        if pool:
            blocks = numpy.array_split(struct, cores)
            tmp_files = pool.map(self._get_block_features, [b for b in blocks if len(b) > 0])
            features = []
            for tmp_file in tqdm(tmp_files, "Restoring block"):
                features += pickle.load(tmp_file)
                os.unlink(tmp_file)
            return features

        cache = {}
        return [self._extract_features(arc, cache) for arc in tqdm(struct, desc="Arc", mininterval=1)]

    def _get_block_features(self, blocks):
        cache = {}
        features = []
        for arc in tqdm(blocks, desc="Block", mininterval=1):
            features.append(self._extract_features(arc, cache))

        _, tmp_file = tempfile.mkstemp(suffix=".pkl")
        pickle.dump(features, tmp_file)

        return tmp_file


def get_corpus(file_name):
    training_corpus = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))
    for doc in training_corpus:
        doc.system_mentions = mention_extractor.extract_system_mentions(doc)

    return training_corpus


def pickle_corpus(file_name, features_file, out_path, skip_if_exists=False):
    extract_substructures = antecedent_trees.extract_substructures_limited
    mention_features, pairwise_features = import_helper.get_features(features_file)

    corpus = get_corpus(file_name)

    InstanceExtractor(extract_substructures,
                      mention_features,
                      pairwise_features,
                      cost_based_on_consistency,
                      out_path,
                      skip_if_exists).extract(corpus)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        file_name = r"D:\ProjetoFinal\data\train\conll\train.conll.train.20"
        features_file = r"D:\GDrive\Puc\Projeto Final\Code\extra_files\features.txt"
        out_path = r"D:\ProjetoFinal\data\pickle"
    else:
        out_path, features_file, file_name = sys.argv[1:]

    pickle_corpus(file_name, features_file, out_path, True)
