import numpy as np
import pandas as pd
import os
import logging

from extension.bert_features import get_doc_df

logging.basicConfig(level=logging.DEBUG)

_bert_db = {}
bins = None  # [-.7, -.5, -.3, -.1, 0, .1, .3, .5, .7]
eye = None  # np.eye(len(bins)+1)
_bert_num_columns = -1  # 300  # 768  # *len(bins)


def load_bins():
    bins_file = os.environ["BINS_FILE"]
    print(f"Using bins file {bins_file}")
    deciles = pd.read_csv(bins_file, header=None)
    bins = deciles.values
    num_classes = bins.shape[0] + 1
    _bert_num_columns = bins.shape[1]
    eye = np.eye(num_classes)
    return bins, eye, _bert_num_columns


def __load_bins():
    global bins
    global eye
    global _bert_num_columns
    bins, eye, _bert_num_columns = load_bins()


def __load_bert_db(doc_id):
    __load_bins()
    global _bert_db

    bert_db_path = os.environ["BERT_DB_PATH"]
    _bert_db[doc_id] = get_doc_df(doc_id, bert_db_path)


def to_classes(embedding):
    def to_bin(pos, x):
        global bins
        for idx, value in enumerate(bins[:, pos]):
            if x < value:
                return idx
        return len(bins)

    classes = [to_bin(idx, x) for idx, x in enumerate(embedding)]
    return np.array(classes)


def to_bins(embedding):
    classes = to_classes(embedding)
    return np.reshape(eye[classes], -1)


def get_embedding(doc_id, begin, end, return_embedding=False, encoder=to_bins):
    if begin >= end:
        return 0
    global _bert_db

    if doc_id not in _bert_db:
        __load_bert_db(doc_id)

    if end > len(_bert_db[doc_id]):
        print(f"Error in {doc_id}. Getting line {end} but has only {len(_bert_db[doc_id])}")

    last_line = _bert_db[doc_id][end - 1, :]
    if begin == 0:
        embedding = last_line / end
        ret_bins = encoder(embedding)
        if return_embedding:
            return ret_bins, list(embedding)
        return ret_bins

    first_line = _bert_db[doc_id][begin - 1, :]
    embedding = (last_line - first_line) / (end - begin)
    ret_bins = encoder(embedding)
    if return_embedding:
        return ret_bins, list(embedding)
    return ret_bins


def bins_between_mentions(anaphor, antecedent):
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

    return "bins_between_mentions", attr


def bins_pair_embedding(anaphor, antecedent):
    _, v1 = bins_embedding(anaphor)
    _, v2 = bins_embedding(antecedent)

    return "bins_pair_embedding", np.concatenate((v2, v1))


def bins_embedding(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.begin
        e = mention.span.end + 1
        return get_embedding(doc_id, b, e)

    return bert_aux("bins_embedding", mention, calc)


def bins_embedding_as_category(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.begin
        e = mention.span.end + 1
        return get_embedding(doc_id, b, e, encoder=to_classes)

    return bert_aux("bins_embedding_as_category", mention, calc)


def bins_last_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.end
        e = mention.span.end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_last_mention", mention, calc)


def bins_middle_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = int((mention.span.begin + mention.span.end) / 2)
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_middle_mention", mention, calc)


def bins_first_mention(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.span.begin
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_first_mention", mention, calc)


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


def bins_first_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].begin
        e = mention.attributes["head_span"].begin + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_first_head", mention, calc)


def bins_last_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].end
        e = mention.attributes["head_span"].end + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_last_head", mention, calc)


def bins_middle_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = int((mention.span.begin + mention.span.end) / 2)
        e = b + 1
        return get_embedding(doc_id, b, e)  # _bert_db.loc[doc_id, :][b:e].mean()

    return bert_aux("bins_middle_head", mention, calc)


def bins_head_embedding(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].begin
        e = mention.attributes["head_span"].end + 1
        return get_embedding(doc_id, b, e, False)

    return bert_aux("bins_head_embedding", mention, calc)


def bins_first_head(mention):
    def calc(mention):
        doc_id = mention.document.identifier
        b = mention.attributes["head_span"].begin
        e = mention.attributes["head_span"].begin + 1
        return get_embedding(doc_id, b, e, False)

    return bert_aux("bins_first_head", mention, calc)


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


def bins_mention_preceding(mention):
    def calc(mention):
        prec = mention.get_context(-1)
        if prec:
            e = mention.span.begin
            b = e - 1
        else:
            b, e = 0, 0

        doc_id = mention.document.identifier

        return get_embedding(doc_id, b, e)

    return bert_aux("bins_mention_preceding", mention, calc)


def bins_mention_governor(mention):
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

    return bert_aux("bins_mention_governor", mention, calc)


def bins_cumsum_between_mentions(mention1, mention2):
    start = mention2.span.end + 1
    end = mention1.span.begin
    multiple_ends = define_cumsum(end - start)
    doc_id = mention1.document.identifier

    embeddings = [get_embedding(doc_id, start, start + x, True) for x in multiple_ends]

    return "bins_cumsum_between_mentions", np.concatenate(embeddings)


def define_cumsum(length):
    return np.cumsum(define_deltas(length))


def define_deltas(length):
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

    return int(length / 5) + np.array(define_deltas(length % 5))
