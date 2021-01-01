import collections
from pre_training.constants import BERT_MODEL_HUB
from cort.core import corpora
import tensorflow as tf
import numpy as np
import random
import os

MAX_MENTION_DISTANCE = 4
MAX_SEQ_LEN = 256
MAX_NUM_PREDICTIONS = 32
MASKED_LM_PROB = 0.10


def create_tokenizer():
    import tensorflow as tf
    import tensorflow_hub as hub
    import bert
    from bert import tokenization

    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def generate_positive_examples(doc, tokenizer):
    for idx, mention in enumerate(doc.annotated_mentions):
        next_mention = None
        c_id = mention.attributes["annotated_set_id"]
        for i in range(1, min(MAX_MENTION_DISTANCE, len(doc.annotated_mentions) - idx)):
            if doc.annotated_mentions[idx + i].attributes["annotated_set_id"] == c_id:
                next_mention = doc.annotated_mentions[idx + i]
                break
        if next_mention is None:
            continue

        yield build_example_from_mentions(mention, next_mention, True, tokenizer)


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def convert_tokens(tokens, seg_id, tokenizer):
    r_tokens = []
    r_seg_ids = []
    for tk, id in zip(tokens, seg_id):
        as_tokens = tokenizer.tokenize(tk)
        r_tokens += as_tokens
        r_seg_ids += [id] * len(as_tokens)

    return r_tokens, r_seg_ids


# only starting words can be masked
def find_possible_positions(tokens):
    return [i for i in range(len(tokens)) if "##" not in tokens[i]]


def discover_range_to_replace(tokens, pos):
    to_replace = [pos]
    for i in range(pos + 1, len(tokens)):
        if tokens[i].startswith("##"):
            to_replace.append(i)
        else:
            break

    return to_replace


def mask_some(tokens, vocab):
    num_to_predict = min(MAX_NUM_PREDICTIONS,
                         max(1, int(round(len(tokens) * MASKED_LM_PROB))))

    candidates = find_possible_positions(tokens)
    masked_positions = []
    masked_tokens = []

    # Same logic as official BERT
    random.shuffle(candidates)
    for pos in candidates:
        if len(masked_positions) >= num_to_predict:
            break

        range_to_replace = discover_range_to_replace(tokens, pos)
        len_to_replace = len(range_to_replace)
        to_replace = None
        if random.random() < 0.8:
            to_replace = ["[MASK]"] * len_to_replace
        elif random.random() < 0.5:
            to_replace = random.choices(vocab, k=len_to_replace)

        if to_replace is not None:
            masked_positions += list(range(pos, pos + len_to_replace))
            masked_tokens += tokens[pos:(pos + len_to_replace)]
            for i in range(len_to_replace):
                tokens[pos + i] = to_replace[i]

    return tokens, masked_positions, masked_tokens


def to_example(tokens, seg_ids, is_coref, tokenizer):
    features = collections.OrderedDict()
    tokens, seg_ids = convert_tokens(tokens, seg_ids, tokenizer)
    tokens, masked_positions, masked_tokens = mask_some(tokens, list(tokenizer.vocab.keys()))
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    masked_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)

    masks = [1] * len(tokens)

    assert len(tokens) <= MAX_SEQ_LEN, f"Too much tokens:{len(tokens)}"

    missing = [0] * (MAX_SEQ_LEN - len(tokens))
    for vec in [tokens, seg_ids, masks]:
        vec += missing

    features["input_ids"] = create_int_feature(tokens)
    features["segment_ids"] = create_int_feature(seg_ids)
    features["input_mask"] = create_int_feature(masks)

    masked_lm_weights = np.ones(len(masked_tokens), dtype=np.int64)

    assert len(masked_lm_weights) <= MAX_NUM_PREDICTIONS, f"Too much predictions:{len(masked_lm_weights)}"

    missing = np.zeros(MAX_NUM_PREDICTIONS - len(masked_lm_weights), dtype=np.int64)
    masked_positions = np.concatenate([masked_positions, missing])
    masked_lm_weights = np.concatenate([masked_lm_weights, missing])
    masked_tokens = np.concatenate([masked_tokens, missing])

    features["masked_lm_positions"] = create_int_feature(masked_positions)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["masked_lm_ids"] = create_int_feature(masked_tokens)

    features["is_coref_labels"] = create_int_feature([is_coref])

    return tf.train.Example(features=tf.train.Features(feature=features))


def build_example_from_mentions(mention_1, mention_2, is_coref, tokenizer):
    doc = mention_1.document
    sentence_1 = doc.sentence_spans[mention_1.attributes["sentence_id"]]
    sentence_2 = doc.sentence_spans[mention_2.attributes["sentence_id"]]

    begin = min(sentence_1.begin, sentence_2.begin)
    end = max(sentence_1.end, sentence_2.end) + 1

    tokens = doc.tokens[begin:end]
    seg_id = [0] * len(tokens)
    for i in range(mention_1.span.begin, mention_1.span.end + 1):
        seg_id[i - begin] = 1

    for i in range(mention_2.span.begin, mention_2.span.end + 1):
        seg_id[i - begin] = 1

    return to_example(tokens, seg_id, is_coref, tokenizer)


def generate_negative_examples(doc, count, tokenizer):
    for idx in random.choices(range(len(doc.annotated_mentions)), k=count):
        mention = doc.annotated_mentions[idx]
        c_id = mention.attributes["annotated_set_id"]

        candidates = doc.annotated_mentions[idx:(idx + MAX_MENTION_DISTANCE)]
        candidates = [x for x in candidates if x.attributes["annotated_set_id"] != c_id]
        if len(candidates) == 0:
            continue
        next_mention = random.choice(candidates)

        yield build_example_from_mentions(mention, next_mention, False, tokenizer)


def process_file(in_file, out_file, tokenizer):
    with open(in_file, encoding="utf-8") as f:
        corpus = corpora.Corpus.from_file("reference", f)

    print(f"{out_file}: ", end='')
    has_mentions = False
    for doc in corpus:
        if len(doc.annotated_mentions) > 0:
            has_mentions = True
            break

    if not has_mentions:
        print("No mentions")
        return

    with tf.io.TFRecordWriter(out_file) as writer:
        for doc in corpus:
            if len(doc.annotated_mentions) == 0:
                continue

            counter = 0
            for example in generate_positive_examples(doc, tokenizer):
                writer.write(example.SerializeToString())
                print("+", end='')
                counter += 2

            for example in generate_negative_examples(doc, counter + 1, tokenizer):
                writer.write(example.SerializeToString())
                print("-", end='')

    print(" end")


def build_files(path_in, path_out):
    print("Creating tokenizer")
    tokenizer = create_tokenizer()

    for r, d, f in os.walk(path_in):
        for file_name in f:
            if not file_name.endswith("_conll"):
                continue

            process_file(os.path.join(r, file_name), f"{path_out}/{file_name}.tsv", tokenizer)


if __name__ == "__main__":
    root = "D:/GDrive/Puc/Projeto Final/Datasets/"

    path_out = f"{root}/finetuning/devel/by_mention_small"
    path_in = f"{root}/conll/development"

    build_files(path_in, path_out)
