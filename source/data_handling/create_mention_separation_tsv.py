import sys
import re
import os
import collections
import numpy as np
import tensorflow as tf
from cort.core import corpora, mention_extractor
from training.train_mentions import MAX_NUM_PREDICTIONS, MAX_NUM_MENTIONS


def linearized_diag(labels, mask, input_size=MAX_NUM_MENTIONS, output_size=MAX_NUM_PREDICTIONS):
    linear_labels = np.zeros(output_size, dtype=int)
    linear_mask = np.zeros(output_size, dtype=int)

    counter = 0
    for offset in range(1, input_size):
        for line in range(input_size - offset):
            linear_labels[counter] = labels[line, line + offset]
            linear_mask[counter] = mask[line, line + offset]
            counter += 1

    return linear_labels, linear_mask


def create_string_feature(value):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, encoding='utf-8')]))
    return f


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


def get_embeddings_for_file(file_name, ret_doc=False, use_annotated=False):
    reference = corpora.Corpus.from_file("reference", open(file_name, "r", encoding="utf-8"))
    for doc in reference:
        mentions = get_mentions(doc, use_annotated)

        if len(mentions) < 2:
            continue
        yield [doc.identifier] + list(get_embeddings_for_mentions(mentions)) + ([doc] if ret_doc else [])


def get_embeddings_for_mentions(mentions):
    n = len(mentions)
    embeddings = []
    labels = np.zeros((n, n))
    mask = np.zeros((n, n))

    for i in range(n):
        mention1 = mentions[i]
        embeddings.append((mention1.span.begin, mention1.span.end))

        for j in range(i + 1, n):
            mention2 = mentions[j]

            labels[i, j] = (mention1.attributes["annotated_set_id"] == mention2.attributes["annotated_set_id"]) and \
                           mention1.attributes["annotated_set_id"] is not None
            mask[i, j] = 1

    return [np.array(embeddings), labels, mask]


def create_ts_file(file_name, out_folder, use_annotated=False):
    for doc_id, embeddings, labels, mask in get_embeddings_for_file(file_name, use_annotated=use_annotated):
        examples = [x for x in to_example(doc_id, embeddings, labels, mask) if x is not None]
        if len(examples) > 0:
            with tf.io.TFRecordWriter(get_file_name(out_folder, doc_id)) as writer:
                for example in examples:
                    writer.write(example.SerializeToString())


def to_example(doc_id, embeddings, labels, mask, ret_features=False):
    if len(embeddings) <= MAX_NUM_MENTIONS:
        for x in to_partial_examples(doc_id, embeddings, labels, mask, ret_features=ret_features):
            yield x
            # yield to_one_example(embeddings, labels, mask, ret_dict)

    for i in range(len(embeddings) - MAX_NUM_MENTIONS):
        j = i + MAX_NUM_MENTIONS
        for x in to_partial_examples(doc_id, embeddings[i:j], labels[i:j, i:j], mask[i:j, i:j],
                                     ret_features=ret_features):
            yield x


def to_partial_examples(doc_id, embeddings, labels, mask, ret_features):
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

    # Checking
    assert labels.shape == (MAX_NUM_MENTIONS, MAX_NUM_MENTIONS), f"Wrong shape: labels {labels.shape}"
    assert mask.shape == (MAX_NUM_MENTIONS, MAX_NUM_MENTIONS), f"Wrong shape: mask {mask.shape}"
    assert embeddings.shape == (MAX_NUM_MENTIONS, 2), f"Wrong shape: embeddings {embeddings.shape}"

    # Input
    features["doc_id"] = create_string_feature(doc_id)
    features["input_embeddings"] = create_int_feature(embeddings.reshape(-1))
    features["input_mask"] = create_int_feature(input_mask)

    # Output
    linear_labels, linear_mask = linearized_diag(labels, mask)

    features["labels"] = create_int_feature(linear_labels)

    off_mask = linear_mask - linear_labels
    off_indices = np.where(off_mask)[0]
    np.random.shuffle(off_indices)

    def build_example_with_mask(effective_mask):
        features["output_mask"] = create_int_feature(effective_mask)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        if sum(linear_labels) == 0:
            example = None

        return [example, features] if ret_features else example

    yield build_example_with_mask(linear_mask)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        in_file = r"D:\GDrive\Puc\Projeto Final\Datasets\conll\train\data\english\annotations\bn\pri\00\pri_0011.gold_conll"
        out_folder = r"D:\ProjetoFinal\data\debug"
    else:
        in_file, out_folder = sys.argv[1:]

    create_ts_file(in_file, out_folder, use_annotated=True)
