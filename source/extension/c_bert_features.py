import numpy as np
import os
from cort.util import import_helper
from extension.bert_features import get_doc_df

_bert_db = {}

_bert_num_columns = None

CUMSUM_SPLITS = 3


def define_deltas(length):
    return define_deltas_3(length)


def __load_bert_db(doc_id):
    bert_db_path = os.environ["BERT_DB_PATH"]
    global _bert_db

    Embedder = import_helper.import_from_path("extension.embedder.BertEmbedder")
    _bert_db[doc_id] = Embedder(get_doc_df(doc_id, bert_db_path))


def get_embedding(doc_id, begin, end):
    global _bert_db
    global _bert_num_columns

    if (doc_id not in _bert_db) or (_bert_num_columns is None):
        __load_bert_db(doc_id)

    return _bert_db[doc_id].get_embedding(begin, end)


def bert_between_mentions(anaphor, antecedent):
    if anaphor.is_dummy():
        b = 0
        e = antecedent.span.begin
    elif antecedent.is_dummy():
        b = 0
        e = anaphor.span.begin
    else:
        b = min(anaphor.span.end, antecedent.span.end)
        e = max(anaphor.span.begin, antecedent.span.begin)

    doc_id = anaphor.document.identifier
    attr = get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return "bert_between_mentions", attr


def bert_pair_embedding(anaphor, antecedent):
    _, v1 = bert_embedding(anaphor)
    _, v2 = bert_embedding(antecedent)

    return "bert_pair_embedding", np.concatenate((v2, v1))


def bert_embedding(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.begin
        e = mention.span.end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_embedding", mention, calc)


def bert_last_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.end
        e = mention.span.end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_last_mention", mention, calc)


def bert_middle_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = int((mention.span.begin + mention.span.end) / 2)
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_last_mention", mention, calc)


def bert_first_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.begin
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_first_mention", mention, calc)


def bert_aux(name, mention, fn):
    if name in mention.attributes:
        attr = mention.attributes[name]
    else:
        if mention.is_dummy():
            attr = np.ones(_bert_num_columns)
        else:
            attr = fn(mention)
        mention.attributes[name] = attr

    return name, attr


def bert_first_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].begin
        e = mention.attributes["head_span"].begin + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_last_head", mention, calc)


def bert_last_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].end
        e = mention.attributes["head_span"].end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_last_head", mention, calc)


def bert_middle_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = int((mention.span.begin + mention.span.end) / 2)
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_middle_head", mention, calc)


def bert_pair_head_embedding(mention1, mention2):
    head1 = bert_head_embedding(mention1)[1]
    head2 = bert_head_embedding(mention2)[1]

    return "bert_pair_head_embedding", (head1 + head2) / 2


def bert_pair_head_dot(mention1, mention2):
    head1 = bert_head_embedding(mention1)[1]
    head2 = bert_head_embedding(mention2)[1]

    return "bert_pair_head_dot", np.inner(head1, head2)


def bert_head_embedding(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].begin
        e = mention.attributes["head_span"].end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bert_head_embedding", mention, calc)


def bert_pair_next(mention1, mention2):
    pair1 = bert_mention_next(mention1)
    pair2 = bert_mention_next(mention2)

    return "bert_pair_next", pair1 + pair2


def bert_mention_next(mention):
    def calc(mention):
        next_t = mention.get_context(1)
        if next_t:
            b = mention.span.end + 1
            e = b + 1
        else:
            b, e = 0, 0

        doc_id = mention.document.identifier

        return get_embedding(doc_id, b, e)

    return bert_aux("bert_mention_next", mention, calc)


def bert_mention_preceding(mention):
    def calc(mention):
        prec = mention.get_context(-1)
        if prec:
            e = mention.span.begin
            b = e - 1
        else:
            b, e = 0, 0

        doc_id = mention.document.identifier

        return get_embedding(doc_id, b, e)

    return bert_aux("bert_mention_preceding", mention, calc)


def bert_mention_governor(mention):
    def calc(mention):
        i = mention.attributes["sentence_id"]
        head_index = mention.attributes["head_index"]
        sentence_span = mention.document.sentence_spans[i]
        span = mention.span

        dep_tree = mention.document.dep[i]

        index = span.begin + head_index - sentence_span.begin

        governor_id = dep_tree[index].head - 1

        if governor_id == -1:
            b, e = 0, 0
        else:
            b = governor_id
            e = b + 1

        doc_id = mention.document.identifier

        return get_embedding(doc_id, b, e)

    return bert_aux("bert_mention_governor", mention, calc)


def bert_cumsum_between_mentions(mention1, mention2):
    doc_id = mention1.document.identifier

    if mention1.is_dummy() or \
            mention2.is_dummy() or \
            mention1.span.embeds(mention2.span) or \
            mention2.span.embeds(mention1.span):
        return "bert_cumsum_between_mentions", np.array([get_embedding(doc_id, 0, 0)] * CUMSUM_SPLITS)

    start = min(mention2.span.end, mention1.span.end) + 1
    end = max(mention2.span.begin, mention1.span.begin)
    multiple_ends = define_cumsum(end - start)

    embeddings = [get_embedding(doc_id, start, start + x) for x in multiple_ends]

    return "bert_cumsum_between_mentions", np.concatenate(embeddings)


def define_cumsum(length):
    return np.cumsum(define_deltas(length))


def define_deltas_5(length):
    if length <= 0:
        return [0, 0, 0, 0, 0]
    if length == 1:
        return [0, 0, 1, 0, 0]
    if length == 2:
        return [0, 1, 0, 1, 0]
    if length == 3:
        return [1, 0, 1, 0, 1]
    if length == 4:
        return [0, 1, 1, 1, 1]

    return int(length / CUMSUM_SPLITS) + np.array(define_deltas_5(length % CUMSUM_SPLITS))


def define_deltas_3(length):
    if length <= 0:
        return [0, 0, 0]
    if length == 1:
        return [0, 1, 0]
    if length == 2:
        return [1, 0, 1]
    if length == 3:
        return [1, 1, 1]

    return int(length / CUMSUM_SPLITS) + np.array(define_deltas_3(length % CUMSUM_SPLITS))
