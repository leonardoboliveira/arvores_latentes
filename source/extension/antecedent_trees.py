import matplotlib.pyplot as plt
import numpy as np
import os
from extension.perceptrons import Perceptron


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


def get_candidate_pairs_old(mentions, max_distance=50, max_distance_with_match=500, debug=False):
    '''
    Yield tuples of mentions, dictionnary of candidate antecedents for the mention
    Arg:
        mentions: an iterator over mention indexes (as returned by get_candidate_mentions)
        max_mention_distance : max distance between a mention and its antecedent
        max_mention_distance_string_match : max distance between a mention and
            its antecedent when there is a proper noun match
    '''
    if debug: print("get_candidate_pairs: mentions", mentions)

    if max_distance_with_match is not None:
        word_to_mentions = {}
        for i in range(len(mentions)):
            if mentions[i].is_dummy():
                continue
            for tok in mentions[i].attributes["tokens"]:
                if not tok in word_to_mentions:
                    word_to_mentions[tok] = [i]
                else:
                    word_to_mentions[tok].append(i)

    for i in range(1, len(mentions)):
        antecedents = set([mentions[k] for k in range(i)]) if max_distance is None \
            else set([mentions[k] for k in range(max(0, i - max_distance), i)])

        antecedents.add(mentions[0])

        if debug: print("antecedents", antecedents)
        if max_distance_with_match is not None:
            for tok in mentions[i].attributes["tokens"]:
                with_string_match = word_to_mentions.get(tok, None)
                for match_idx in with_string_match:
                    if (match_idx < i) and (match_idx >= (i - max_distance_with_match)):
                        antecedents.add(mentions[match_idx])

        yield i, sorted(list(antecedents), key=lambda x: 0 if x.span is None else x.span.begin, reverse=True)


class AntecedentTreePerceptron(Perceptron):
    """ A perceptron for antecedent trees. """
    first_doc = None
    boost = 1
    perf = []

    def __init__(self,
                 n_iter=5,
                 seed=23,
                 cost_scaling=1,
                 priors=None,
                 weights=None,
                 cluster_features=None):
        if isinstance(priors, int):
            self.counter = priors
            self.cached_weights = weights[1]
            weights = weights[0]
        else:
            self.cached_weights = {}
            self.counter = 0
            
        Perceptron.__init__(self, n_iter, seed, cost_scaling, None, weights, cluster_features)

    def fit(self, substructures, arc_information):
        x = super().fit(substructures, arc_information)
        # plt.hist(self.perf)
        # plt.savefig("performance.png")
        # plt.show()
        # print(f"Counter:{self.counter}")
        return x

    def predict(self, substructures, arc_information):
        x = super().predict(substructures, arc_information)
        # plt.hist(self.perf)
        # plt.savefig("performance_predict.png")
        # plt.show()
        print(f"Performance Predict:{np.mean(self.perf)} +- {np.std(self.perf)} [{len(self.perf)}]")
        return x

    def argmax(self, substructure, arc_information):
        """ Decoder for antecedent trees.

        Compute highest-scoring antecedent tree and highest-scoring antecedent
        tree consistent with the gold annotation.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
                pairs in the following order:
                (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.

        Returns:
            A 7-tuple describing the highest-scoring antecedent tree, and the
            highest-scoring antecedent tree consistent with the gold
            annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree,
                - **best_cons_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree consistent
                  with the gold annotation.
                - **best_cons_labels** (*list(str)*): empty, the antecedent
                  tree approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree consistent with
                  the gold annotation,
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent tree is consistent with the gold annotation.
        """
        if not substructure:
            return [], [], [], [], [], [], True

        if len(substructure) > 0:
            doc_id = substructure[0][0].document.identifier
            if self.first_doc is None:
                self.first_doc = doc_id
            elif doc_id == self.first_doc:
                self.boost += 1
        arcs = []
        arcs_scores = []
        coref_arcs = []
        coref_arcs_scores = []

        is_consistent = True
        count_inconsistent = 0
        count_consistent = 0
        for sub_tree in group_arcs(substructure):
            best, max_val, best_cons, max_cons, best_is_consistent = \
                self.find_best_arcs(sub_tree, arc_information)

            if best is not None:
                arcs.append(best)
                arcs_scores.append(max_val)
            else:
                pass  # print("No best")

            if best_cons is not None:
                coref_arcs.append(best_cons)
                coref_arcs_scores.append(max_cons)
            else:
                pass  # print("No best")

            is_consistent &= best_is_consistent
            if best is None:
                if self.has_any_consistent(sub_tree, arc_information):
                    count_inconsistent += 1
                else:
                    count_consistent += 1
            else:
                if best[1].is_dummy():
                    if best[0].attributes["first_in_gold_entity"]:
                        count_consistent += 1
                    else:
                        count_inconsistent += 1
                else:
                    if best_is_consistent:
                        count_consistent += 1
                    else:
                        count_inconsistent += 1

        if (count_consistent + count_inconsistent) > 0:
            self.perf.append(count_consistent / (count_consistent + count_inconsistent))

        return (
            arcs,
            [],
            arcs_scores,
            coref_arcs,
            [],
            coref_arcs_scores,
            is_consistent
        )

    def has_any_consistent(self, sub_tree, arc_information):
        for arc in sub_tree:
            features, costs, consistent = arc_information[arc]
            if consistent:
                return True
        return False


class AntecedentTreePerceptronWithMargin(AntecedentTreePerceptron):

    def find_best_arcs(self, arcs, arc_information, label="+"):
        """ Find the highest-scoring arc and arc consistent with the gold
        information among a set of arcs.

        Args:
            arcs (list((Mention, Mention)): A list of mention pairs constituting
                arcs.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
            label (str): The label of the arcs. Defaults to "+".

        Returns:
            A 5-tuple describing the highest-scoring anaphor-antecedent
            decision, and the highest-scoring anaphor-antecedent decision
            consistent with the gold annotation. The tuple consists of:

                - **best** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision.
                - **max_val** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision,
                - **best_cons** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **max_const** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **is_consistent** (*bool*): whether the highest-scoring
                  anaphor-antecedent decision is consistent with the gold
                  information.
        """
        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False
        first_ana, first_ante = arcs[0]
        for arc in arcs:
            features, costs, consistent = arc_information[arc]
            assert arc[0] == first_ana, f"Not correct node expected {first_ana} but was {arc[0]}"
            nonnumeric_features, numeric_features, numeric_vals = features

            score = self.score_arc(arc, arc_information, label)

            if not consistent:
                score += 2000
            elif arc[1].is_dummy() and not arc[0].attributes["first_in_gold_entity"]:
                score += 3500

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = consistent

            if score > max_cons and consistent:
                best_cons = arc
                max_cons = score

        return best, max_val, best_cons, max_cons, best_is_consistent


class AntecedentTreePerceptronNoAdjust(AntecedentTreePerceptron):
    def __init__(self,
                 n_iter=5,
                 seed=23,
                 cost_scaling=1,
                 priors=None,
                 weights=None,
                 cluster_features=None):
        AntecedentTreePerceptron.__init__(self, n_iter, seed, cost_scaling, None, weights, cluster_features)
