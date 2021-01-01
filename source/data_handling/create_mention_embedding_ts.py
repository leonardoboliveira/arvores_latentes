from cort.core import corpora, mention_extractor
import collections
from extension import bert_features
import tensorflow as tf
import numpy as np
import re
import os
import sys
from training.train_mentions import MAX_NUM_PREDICTIONS, MAX_NUM_MENTIONS


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def get_file_name(out_folder, doc_id):
    name_re = re.compile(r".*\((.*)\); part (\d\d\d)")
    match = name_re.match(doc_id)
    file_name = f"{match.group(1)}_{match.group(2)}".replace("/", "_")
    final_name = os.path.join(out_folder, file_name + ".tsv")
    # print(f"##### {final_name}")
    return final_name


def create_ts_file(file_name, out_folder, use_annotated=False, as_diag=False):
    for doc_id, embeddings, labels, mask, cumsum in get_embeddings_for_file(file_name, use_annotated=use_annotated):
        examples = [x for x in to_example(embeddings, labels, mask, cumsum, as_diag=as_diag) if x is not None]
        if len(examples) > 0:
            with tf.io.TFRecordWriter(get_file_name(out_folder, doc_id)) as writer:
                for example in examples:
                    writer.write(example.SerializeToString())


def to_example(embeddings, labels, mask, cumsum=None, ret_dict=False, num_examples=10, as_diag=False):
    if len(embeddings) <= MAX_NUM_MENTIONS:
        for x in to_partial_examples(embeddings, labels, mask, cumsum,
                                     ret_dict=ret_dict,
                                     num_examples=num_examples,
                                     as_diag=as_diag):
            yield x
        # yield to_one_example(embeddings, labels, mask, ret_dict)

    for i in range(len(embeddings) - MAX_NUM_MENTIONS):
        j = i + MAX_NUM_MENTIONS
        for x in to_partial_examples(embeddings[i:j], labels[i:j, i:j], mask[i:j, i:j],
                                     cumsum[i:j, i:j] if (cumsum is not None) else None,
                                     ret_dict=ret_dict,
                                     num_examples=num_examples,
                                     as_diag=as_diag):
            yield x
        # yield to_one_example(embeddings[i:j], labels[i:j, i:j], mask[i:j, i:j], ret_dict)


def linearized_diag(labels, mask, cumsum, input_size=MAX_NUM_MENTIONS, output_size=MAX_NUM_PREDICTIONS):
    linear_labels = np.zeros(output_size, dtype=int)
    linear_mask = np.zeros(output_size, dtype=int)
    linear_cumsum = np.zeros((output_size, 5, cumsum.shape[-1] if (cumsum is not None) else 0))

    counter = 0
    for offset in range(1, input_size):
        for line in range(input_size - offset):
            linear_labels[counter] = labels[line, line + offset]
            linear_mask[counter] = mask[line, line + offset]
            if cumsum is not None:
                linear_cumsum[counter] = cumsum[line, line + offset]
            counter += 1

    return linear_labels, linear_mask, np.reshape(linear_cumsum, -1)


def linearized_horizontal(labels, mask, cumsum):
    linear_labels = np.zeros(MAX_NUM_PREDICTIONS, dtype=int)
    linear_mask = np.zeros(MAX_NUM_PREDICTIONS, dtype=int)
    linear_cumsum = np.zeros((MAX_NUM_PREDICTIONS, 5, cumsum.shape[-1]))

    counter = 0
    for i in range(MAX_NUM_MENTIONS):
        for j in range(i + 1, MAX_NUM_MENTIONS):
            linear_labels[counter] = labels[i, j]
            linear_mask[counter] = mask[i, j]
            if cumsum is not None:
                linear_cumsum[counter] = cumsum[i, j]
            counter += 1

    return linear_labels, linear_mask, np.reshape(linear_cumsum, -1)


def to_partial_examples(embeddings, labels, mask, cumsum, ret_dict=False, num_examples=10, as_diag=False):
    features = collections.OrderedDict()

    missing_inputs = MAX_NUM_MENTIONS - embeddings.shape[0]
    assert missing_inputs >= 0, f"Too many mentions {embeddings.shape}"
    assert len(embeddings.shape) == 2, f"Strange shape:{embeddings.shape}"

    input_mask = np.ones(embeddings.shape[0], dtype=int)

    if missing_inputs > 0:
        embeddings = np.pad(embeddings, ((0, missing_inputs), (0, 0)))
        labels = np.pad(labels, ((0, missing_inputs), (0, missing_inputs)))
        mask = np.pad(mask, ((0, missing_inputs), (0, missing_inputs)))
        input_mask = np.pad(input_mask, ((0, missing_inputs)))
        if cumsum is not None:
            cumsum = np.pad(cumsum, ((0, missing_inputs), (0, missing_inputs), (0, 0), (0, 0)))

    # Checking
    assert labels.shape == (MAX_NUM_MENTIONS, MAX_NUM_MENTIONS), f"Wrong shape: labels {labels.shape}"
    assert mask.shape == (MAX_NUM_MENTIONS, MAX_NUM_MENTIONS), f"Wrong shape: mask {mask.shape}"
    assert embeddings.shape == (MAX_NUM_MENTIONS, 768), f"Wrong shape: embeddings {embeddings.shape}"
    if cumsum is not None:
        assert cumsum.shape == (MAX_NUM_MENTIONS, MAX_NUM_MENTIONS, 5, 768), f"Wrong shape: cumsum {cumsum.shape}"

    # Input
    features["input_embeddings"] = create_float_feature(embeddings.reshape(-1))
    features["input_mask"] = create_int_feature(input_mask)

    # Output
    if as_diag:
        linear_labels, linear_mask, linear_cumsum = linearized_diag(labels, mask, cumsum)
    else:
        linear_labels, linear_mask, linear_cumsum = linearized_horizontal(labels, mask, cumsum)

    features["labels"] = create_int_feature(linear_labels)
    if cumsum is not None:
        assert linear_cumsum.shape == (MAX_NUM_PREDICTIONS * 5 * 768,), f"Wrong shape {linear_cumsum.shape}"
        features["cumsum"] = create_float_feature(linear_cumsum)

    off_mask = linear_mask - linear_labels
    off_indices = np.where(off_mask)[0]
    np.random.shuffle(off_indices)
    num_positive = np.sum(linear_labels)

    def build_example_with_mask(effective_mask):
        features["output_mask"] = create_int_feature(effective_mask)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        if sum(linear_labels) == 0:
            example = None
        return [example, features] if ret_dict else example

    yield build_example_with_mask(linear_mask)


"""
    returned_something = False
    for group_index in list(zip(*[iter(off_indices)] * num_positive))[:num_examples]:
        effective_mask = np.zeros(MAX_NUM_PREDICTIONS, dtype=int)
        effective_mask[sorted(group_index)] = 1
        effective_mask += linear_labels

        assert max(effective_mask) == 1

        yield build_example_with_mask(effective_mask)
        returned_something = True

    if not returned_something:
        yield build_example_with_mask(linear_labels)
"""


def get_embeddings_for_file(file_name, ret_doc=False, use_annotated=False):
    reference = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))
    for doc in reference:
        mentions = get_mentions(doc, use_annotated)

        if len(mentions) < 2:
            continue
        bert_features.get_embedding(doc.identifier, 0, 1)
        yield [doc.identifier] + list(get_embeddings_for_mentions(mentions)) + ([doc] if ret_doc else [])


def get_mentions(doc, use_annotated):
    if not doc.system_mentions:
        mentions = mention_extractor.extract_system_mentions(doc)
    else:
        mentions = doc.system_mentions

    if use_annotated:
        mentions += doc.annotated_mentions
    mentions = [m for m in set(mentions) if not m.is_dummy()]
    mentions = sorted(mentions, key=lambda x: (x.span.begin, -x.span.end))
    return mentions


def get_embeddings_for_mentions(mentions, ignore_cumsum=False):
    n = len(mentions)
    embeddings = []
    labels = np.zeros((n, n))
    mask = np.zeros((n, n))
    cumsum = None

    for i in range(n):
        mention1 = mentions[i]
        embeddings.append(bert_features.bert_embedding(mention1)[1])

        for j in range(i + 1, n):
            mention2 = mentions[j]
            if not ignore_cumsum:
                cumsum_emb = bert_features.bert_cumsum_between_mentions(mention1, mention2)[1]
                cumsum_emb = cumsum_emb.reshape((5, -1))
                if cumsum is None:
                    # Creating a tensor of n x n x 5 x 768 (A matrix with a embedding)
                    cumsum = np.zeros([n, n] + list(cumsum_emb.shape))
                cumsum[i, j] = cumsum_emb

            labels[i, j] = (mention1.attributes["annotated_set_id"] == mention2.attributes["annotated_set_id"]) and \
                           mention1.attributes["annotated_set_id"] is not None
            mask[i, j] = 1

    return [np.array(embeddings), labels, mask] + ([] if ignore_cumsum else [cumsum])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        in_file = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train\data\english\annotations\bn\pri\00\pri_0076.gold_conll"
        out_folder = r"D:\ProjetoFinal\data\debug"
    else:
        in_file, out_folder = sys.argv[1:]

    create_ts_file(in_file, out_folder, use_annotated=True, as_diag=True)
    # import matplotlib.pyplot as plt
    # plt.hist(useful)
    # plt.show()
