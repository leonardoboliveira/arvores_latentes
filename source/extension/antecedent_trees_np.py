import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import norm
from tqdm import tqdm
import random
import multiprocessing

FEATURE_DIM = 2 ** 24


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def to_sparse_vector(features):
    nonnumeric_features, numeric_features, numeric_vals = features
    indexes = np.concatenate([nonnumeric_features, numeric_features])
    data = np.concatenate([[1] * len(nonnumeric_features), numeric_vals])

    return indexes, data


def group_arcs(substructure):
    ana = None
    arcs = []
    for arc in substructure:
        if ana == arc[0]:
            arcs.append(arc)
            continue
        if len(arcs) > 0:
            yield arcs

        ana = arc[0]
        arcs = [arc]

    if len(arcs) > 0:
        yield arcs


def extract_substructures_limited(doc):
    """ Extract the search space for the antecedent tree model,

    The mention ranking model consists in computing the optimal antecedent for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the tree.

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The nested list contains
    only one list, since antecedent trees have only one substructure for
    each document.

    The list contains all potential (anaphor, antecedent) pairs in the
    following order: (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...,
    where m_j is the jth mention in the document.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructure = []
    for i, antecedents in get_candidate_pairs(doc.system_mentions, int(os.environ["MAX_DISTANCE"])):
        ana = doc.system_mentions[i]
        for ante in antecedents:
            assert not ana.is_dummy()
            substructure.append((ana, ante))

    return [substructure]


def get_candidate_pairs(mentions, max_distance=50):
    for idx1, m1 in enumerate(mentions):
        if idx1 == 0:
            continue

        first_index = max(0, idx1 - max_distance)
        candidates = mentions[first_index:idx1]
        candidates += list(get_candidates_for_mention(m1, idx1, mentions[:first_index]))
        if len(candidates) > 0:
            # yield idx1, sorted(list(candidates), key=lambda x: 0 if x.span is None else x.span.begin, reverse=True)
            yield idx1, sorted(candidates, reverse=True)


def is_same_ne(mention1, mention2):
    return False


def head_word_match(mention1, mention2):
    if mention1.is_dummy() or mention2.is_dummy():
        return True

    h1 = " ".join(mention1.attributes["head"])
    h2 = " ".join(mention2.attributes["head"])
    return h1 == h2


def shallow_discourse(mention1, mention2):
    return False


def match_pronoun(mention1, mention2):
    # Missing animacy
    return mention1.attributes["gender"] == mention2.attributes["gender"] \
           and mention1.attributes["number"] == mention2.attributes["gender"] \
           and mention1.attributes["speaker"] == mention2.attributes["speaker"]


def pronoun_and_ne(mention1, mention2):
    return False


def get_candidates_for_mention(mention, idx, mentions):
    def is_candidate(ana):
        return is_same_ne(mention, ana) \
               or head_word_match(mention, ana) \
               or shallow_discourse(mention, ana) \
               or match_pronoun(mention, ana) \
               or pronoun_and_ne(mention, ana)

    return [x for x in mentions if is_candidate(x)]


def analyze_substructure(arc_information, substructure, weights, margin):
    temp_weights = csc_matrix((FEATURE_DIM, 1))
    is_consistent = True
    ok = 0
    nok = 0
    groups = list(group_arcs(substructure))

    for sub_tree in tqdm(groups, "Arc Group"):
        data = find_best_arc(sub_tree, arc_information, weights, margin)
        best_arc, best_features, best_coref_arc, best_coref_features, best_is_consistent = data
        if (best_arc[1] != best_coref_arc[1]) or (best_arc[0] != best_coref_arc[0]):
            temp_weights += best_coref_features - best_features
            nok += 1
        else:
            ok += 1

        is_consistent &= best_is_consistent
    return is_consistent, temp_weights, ok, nok


def find_best_arc(sub_tree, arc_information, weights, margin):
    row_ind = []
    col_ind = []
    data = []
    consistent = []
    artificial = []

    for idx, arc in enumerate(sub_tree):
        features, costs, this_consistent = arc_information[arc]
        indexes, this_data = to_sparse_vector(features)
        col_ind += [idx] * len(indexes)
        row_ind.append(indexes)
        data.append(this_data)
        if arc[1].is_dummy():
            artificial.append(1)
            consistent.append(arc[0].attributes["first_in_gold_entity"])
        else:
            artificial.append(0)
            consistent.append(this_consistent)

    row_ind = np.concatenate(row_ind)
    data = np.concatenate(data)
    consistent = np.array(consistent)
    artificial = np.array(artificial)

    feature_matrix = csc_matrix((data, (row_ind, col_ind)), (FEATURE_DIM, len(sub_tree)))

    all_scores = np.dot(weights, feature_matrix).toarray()
    best_consistent_row = np.argmax(all_scores * (consistent - artificial) - (1 - consistent) * 10000)
    loss = (1 - consistent) * (1 + artificial * 1.5)

    best_row = np.argmax(all_scores + margin * loss)

    return sub_tree[best_row], \
           feature_matrix[:, best_row], \
           sub_tree[best_consistent_row], \
           feature_matrix[:, best_consistent_row], \
           consistent[best_row]


def get_chunk_data(chunk, arc_information, substructures, weights, margin):
    with multiprocessing.Pool() as pool:
        data = pool.map(MyRunner(arc_information, substructures, weights, margin), tqdm(chunk, desc="Chunk"))
    output = []
    t_ok = 0
    t_nok = 0
    for weights, ok, nok in data:
        output.append(weights)
        t_ok += ok
        t_nok += nok
    print(f"Acc Doc Perf:{t_ok / (t_ok + t_nok)}")
    return output


class MyRunner:
    def __init__(self, arc_information, substructures, weights, margin):
        self.arc_information = arc_information
        self.substructures = substructures
        self.weights = weights
        self.margin = margin

    def __call__(self, idx):
        substructure = self.substructures[idx]
        is_consistent, temp_weights, ok, nok = analyze_substructure(self.arc_information, substructure, self.weights,
                                                                    self.margin)
        if not is_consistent:
            return temp_weights, ok, nok
        return None, ok, nok


class AntecedentTreePerceptron():
    def __init__(self, weights=None, priors=0, n_iter=5, cost_scaling=None, seed=0):
        random.seed(seed)

        if weights:
            self.weights = weights
        else:
            self.weights = csr_matrix((1, FEATURE_DIM))

        if priors:
            self.counter = priors
        else:
            self.counter = 0

        self.n_iter = n_iter
        self.margin = 2000

    def fit(self, substructures, arc_information):
        indexes = list(range(0, len(substructures)))
        for epoch in range(self.n_iter):
            print(f"Starting epoch {epoch}")

            random.shuffle(indexes)
            errors = 0
            clusters = 24
            chunks = divide_chunks(indexes, clusters)
            for chunk in tqdm(chunks, desc="Training Chunks"):
                if len(chunk) == 0:
                    continue
                for weights in get_chunk_data(chunk, arc_information, substructures, self.weights, self.margin):
                    if weights is not None:
                        self.weights += weights.transpose()
                        errors += 1
                    self.counter += 1

            print(f"Errors: {errors}/{len(indexes)}")

    def update_weights(self, arcs_features, coref_features):
        self.weights += np.sum(coref_features) - np.sum(arcs_features)

    def predict(self, substructures, arc_information):
        x = super().predict(substructures, arc_information)
        plt.hist(self.perf)
        plt.savefig("performance_predict.png")
        plt.show()
        print(f"Performance Predict:{np.mean(self.perf)} +- {np.std(self.perf)}")
        return x

    def get_model(self):
        return self.counter, self.weights

    def get_labels(self):
        return ["+"]
