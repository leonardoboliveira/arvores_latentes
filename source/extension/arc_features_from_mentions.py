from data_handling.create_mention_separation_tsv import get_embeddings_for_mentions, to_example, MAX_NUM_MENTIONS, \
    get_mentions, MAX_NUM_PREDICTIONS
from training.train_mentions import get_model, INPUT_EMBEDDINGS_SIZE, cumsum_embedding, mention_embedding

import os
import numpy as np

_arc_db = {}
_encoder = None
ENCODER_MODEL = os.environ.get("ENCODER_MODEL", None)


def create_mask(input):
    output = np.zeros(MAX_NUM_PREDICTIONS)
    labels = (input.sum(axis=1) != 0)
    counter = 0
    for offset in range(1, MAX_NUM_MENTIONS):
        for line in range(MAX_NUM_MENTIONS - offset):
            output[counter] = labels[line] * labels[line + offset]
            counter += 1

    return output.reshape((MAX_NUM_PREDICTIONS, 1))


def create_fake_labels(labels):
    fake = np.zeros(shape=labels.shape)
    for i in range(0, int(fake.shape[0])):
        for j in range(i + 1, fake.shape[1]):
            fake[i, j] = (i + 1) % 2

    return fake


def __load_encoder():
    global _encoder

    _, _encoder = get_model()
    _encoder.load_weights(ENCODER_MODEL)


def arc_index(i, j):
    return int((i * MAX_NUM_MENTIONS + j) - (1 + i + 1) * (i + 1) / 2)


def arc_index_diag(i, j):
    offset = j - i
    K = offset - 1
    pass_before = (2 * MAX_NUM_MENTIONS - 1 - K) * K / 2
    return int(pass_before) + i


def example_features_to_model_features(example, input_embeddings_size, max_num_mentions, max_num_predictions,
                                       ignore_cumsum):
    spans = np.reshape(example["input_embeddings"].int64_list.value, (max_num_mentions, 2))
    doc_id = example["doc_id"].bytes_list.value
    embeddings = mention_embedding(doc_id, spans)
    # tf.print("Example: DocId", example["doc_id"], "Span:", spans[0], "Embed:", embeddings[0])
    mask = np.array(example["output_mask"].int64_list.value)
    input_map = {"output_mask": mask}
    if not ignore_cumsum:
        cumsum = cumsum_embedding(doc_id, spans, max_num_predictions, input_embeddings_size)
        input_map["cumsum"] = cumsum
    input_map["input_embeddings"] = np.reshape(embeddings, (max_num_mentions, input_embeddings_size))
    return input_map


def get_predictions(doc, model, use_mask=False, use_cumsum=False):
    mentions = get_mentions(doc, False)

    if len(mentions) < 2:
        return {}

    all_embeddings = []
    all_masks = []
    spans = []
    cumsum = []

    for i in range(len(mentions)):
        mentions[i].id = i

    embeddings, labels, mask = get_embeddings_for_mentions(mentions)
    labels = create_fake_labels(labels)
    counter = 0

    for example, features in to_example(doc.identifier, embeddings, labels, mask, ret_features=True):
        input_map = example_features_to_model_features(features, INPUT_EMBEDDINGS_SIZE, MAX_NUM_MENTIONS,
                                                       MAX_NUM_PREDICTIONS, ignore_cumsum=not use_cumsum)
        all_embeddings.append(input_map["input_embeddings"])
        output_mask = create_mask(input_map["input_embeddings"])
        all_masks.append(output_mask)
        if use_cumsum:
            cumsum.append(input_map["cumsum"])
        length = sum(features["input_mask"].int64_list.value)
        spans.append((counter, length))
        counter += 1

    if len(all_embeddings) == 0:
        print("No embeddings")

    params = [np.array(all_embeddings)]
    if use_mask:
        params.append(np.array(all_masks))
    if use_cumsum:
        params.append(np.array(cumsum))

    all_arc_embeddings = model.predict(params)

    def distribute_into_slices(all_arc_embeddings):
        arc_all_info = {}
        for start, length in spans:
            mention1 = mentions[start]
            for i in range(length - 1):  # -1 because the first mention is already defined
                mention2 = mentions[start + i + 1]
                key = (mention1, mention2)
                if key not in arc_all_info:
                    arc_all_info[key] = []

                arc_idx = arc_index_diag(mention1.id - start, mention2.id - start)

                if arc_idx >= all_arc_embeddings.shape[1]:
                    print("Too big")

                arc_all_info[key].append(
                    (start, start + length - 1, all_arc_embeddings[start, arc_idx, :]))
        return arc_all_info

    arc_all_info = distribute_into_slices(all_arc_embeddings)

    def choose_best_slice(arc_all_info):
        arc_info = {}
        for arc, infos in arc_all_info.items():
            score = -1
            mention1, mention2 = arc
            for start, end, arc_embedding in infos:
                left_context = min(mention1.id, mention2.id) - start
                right_context = end - max(mention1.id, mention2.id)

                this_score = min(left_context, right_context)
                if this_score > score:
                    score = this_score
                    arc_info[arc] = arc_embedding
        return arc_info

    arc_info = choose_best_slice(arc_all_info)

    return arc_info


def _get_arc_embedding(doc):
    global _encoder
    if _encoder is None:
        __load_encoder()

    return get_predictions(doc, _encoder, use_cumsum=True)


def bert_arc_embedding(mention1, mention2):
    global _arc_db
    doc_id = mention1.document.identifier
    if doc_id not in _arc_db:
        _arc_db[doc_id] = _get_arc_embedding(mention1.document)

    if not hasattr(mention1, "id") or not hasattr(mention1, "id"):
        print(f"Not found attribute for mentions in {doc_id}")
        return "bert_arc_embedding", []

    if mention1.id < mention2.id:
        arc = (mention1, mention2)
    else:
        arc = (mention2, mention1)

    if arc not in _arc_db[doc_id]:
        return "bert_arc_embedding", []

    return "bert_arc_embedding", _arc_db[doc_id][arc]
