from __future__ import absolute_import, division, print_function, unicode_literals
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import multiprocessing
from training.train_mentions_model import get_model, MAX_NUM_PREDICTIONS, INPUT_EMBEDDINGS_SIZE, MAX_NUM_MENTIONS
from extension.bert_features import get_embedding, define_cumsum, CUMSUM_SPLITS

BUFFER_SIZE = 100
EPOCHS = 100
VALIDATION_STEPS = 200
BATCH_SIZE = 8
VALIDATIONS_PER_EPOCH = 5
REDUCTIONS_STEPS = 5

project_id = 'topicosmultimedia'


def input_fn_builder(input_files,
                     is_training,
                     decode_record_fn=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    if decode_record_fn is None:
        decode_record_fn = _decode_record

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        buffer_size = params["buffer_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            options = tf.data.Options()
            options.experimental_deterministic = False
            d = d.with_options(options).interleave(lambda x: tf.data.TFRecordDataset(x))
            d = d.shuffle(buffer_size=buffer_size)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.map(lambda record: decode_record_fn(record, params))
        d = d.batch(batch_size=batch_size, drop_remainder=True)

        return d

    return input_fn


def reduce_mask(mask, labels, reduction):
    off_mask = mask - labels
    rand_off = np.random.rand(*mask.shape) > reduction
    return (rand_off * off_mask) + labels


def mention_embedding(doc_id, mentions):
    doc_id = doc_id[0].decode("utf-8")

    get_embedding(doc_id, 0, 0)
    embeddings = []
    for i in range(mentions.shape[0]):
        df = get_embedding(doc_id, mentions[i, 0], mentions[i, 1] + 1)
        df = df.values
        if len(df.shape) > 1:
            df = df.reshape(-1)
        assert df.shape[0] == INPUT_EMBEDDINGS_SIZE, \
            f"Wrong shape:{df.shape}, {type(df)}, {mentions[i, 0]}, {mentions[i, 1]} {doc_id}"
        embeddings.append(df)

    out = np.array(embeddings)
    assert out.shape == (len(mentions), INPUT_EMBEDDINGS_SIZE), f"Wrong shape : {out.shape}, {doc_id}"
    return np.reshape(out, (len(mentions), INPUT_EMBEDDINGS_SIZE))


def cumsum_embedding(doc_id, spans, max_num_predictions, input_embeddings_size):
    doc_id = doc_id[0].decode("utf-8")

    def embeds(m1, m2):
        return (m1[0] <= m2[1]) and (m1[0] >= m2[1])

    def cumsum(mention1, mention2):
        if embeds(mention1, mention2) or embeds(mention2, mention1):
            return np.array([get_embedding(doc_id, 0, 0)] * CUMSUM_SPLITS)
        start = min(mention2[1], mention1[1]) + 1
        end = max(mention2[0], mention1[0])
        multiple_ends = define_cumsum(end - start)

        embeddings = [get_embedding(doc_id, start, start + x) for x in multiple_ends]

        return pd.concat(embeddings, axis=1).transpose().values

    input_size = len(spans)
    linear_cumsum = np.zeros((max_num_predictions, CUMSUM_SPLITS, input_embeddings_size))
    counter = 0
    for offset in range(1, input_size):
        for line in range(input_size - offset):
            linear_cumsum[counter] = cumsum(spans[line], spans[line + offset])
            counter += 1

    return linear_cumsum


def _from_mention_to_record(record, params):
    max_num_mentions = params["max_num_mentions"]
    max_num_predictions = params["max_num_predictions"]
    reduction = params["reduction"]
    input_embeddings_size = params["input_embeddings_size"]

    name_to_features = {
        "input_embeddings":
            tf.io.FixedLenFeature(max_num_mentions * 2, tf.int64),
        "input_mask":
            tf.io.FixedLenFeature(max_num_mentions, tf.int64),
        "labels":
            tf.io.FixedLenFeature(max_num_predictions, tf.int64),
        "output_mask":
            tf.io.FixedLenFeature(max_num_predictions, tf.int64),
        "doc_id":
            tf.io.FixedLenFeature(1, tf.string),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    spans = tf.reshape(example["input_embeddings"], (max_num_mentions, 2))
    embeddings = tf.numpy_function(mention_embedding, [example["doc_id"], spans], [tf.double])

    # tf.print("Example: DocId", example["doc_id"], "Span:", spans[0], "Embed:", embeddings[0])
    mask = tf.numpy_function(reduce_mask, [example["output_mask"], example["labels"], reduction], [tf.int32])
    mask = tf.cast(tf.reshape(mask, (max_num_predictions, 1)), tf.float32)

    Y = tf.one_hot(example["labels"], depth=2)
    Y = tf.multiply(Y, mask)

    input_map = {"output_mask": mask}

    if not params.get("ignore_cumsum", False):
        cumsum = tf.numpy_function(cumsum_embedding,
                                   [example["doc_id"], spans, max_num_predictions, input_embeddings_size],
                                   [tf.double])
        cumsum = tf.reshape(cumsum, (max_num_predictions, CUMSUM_SPLITS, input_embeddings_size))
        input_map["cumsum"] = cumsum

    # tf.print("Embeddings:", embeddings[0])
    input_map["input_embeddings"] = tf.reshape(embeddings, (max_num_mentions, input_embeddings_size))
    return input_map, Y


# @tf.function
def _decode_record(record, params):
    max_num_mentions = params["max_num_mentions"]
    input_embeddings_size = params["input_embeddings_size"]
    max_num_predictions = params["max_num_predictions"]

    name_to_features = {
        "input_embeddings":
            tf.io.FixedLenFeature(max_num_mentions * input_embeddings_size, tf.float32),
        "input_mask":
            tf.io.FixedLenFeature(max_num_mentions, tf.int64),
        "labels":
            tf.io.FixedLenFeature(max_num_predictions, tf.int64),
        "output_mask":
            tf.io.FixedLenFeature(max_num_predictions, tf.int64),
    }
    if not params.get("ignore_cumsum", False):
        name_to_features["cumsum"] = tf.io.FixedLenFeature(max_num_predictions * input_embeddings_size * CUMSUM_SPLITS,
                                                           tf.float32),

    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)
    reduction = params["reduction"]

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    embeddings = tf.reshape(example["input_embeddings"], (max_num_mentions, input_embeddings_size))
    mask = tf.numpy_function(reduce_mask, [example["output_mask"], example["labels"], reduction], [tf.int32])
    mask = tf.cast(tf.reshape(mask, (max_num_predictions, 1)), tf.float32)

    Y = tf.one_hot(example["labels"], depth=2)
    Y = tf.multiply(Y, mask)

    input_map = {"input_embeddings": embeddings,
                 "output_mask": mask}

    if not params.get("ignore_cumsum", False):
        input_map["cumsum"] = tf.reshape(example["cumsum"], (max_num_predictions, CUMSUM_SPLITS, input_embeddings_size))

    return input_map, Y


def slice_and_split_input(data_folder):
    if isinstance(data_folder, str):
        files = get_files(data_folder)
        if len(files) == 0:
            print(f"No files found in folder {data_folder}")
            return
        train_files, valid_files = train_test_split(files)
        print(f"Using {len(train_files)} files for train and {len(valid_files)} for validation")
    else:
        # Two folders -> train/test already splitted
        train_files = os.listdir(data_folder[0])
        train_files = [f"{data_folder}/{f}" for f in train_files if f.endswith(".tsv")]

        valid_files = os.listdir(data_folder[1])
        valid_files = [f"{data_folder}/{f}" for f in valid_files if f.endswith(".tsv")]

    train_input = input_fn_builder(train_files, True, _from_mention_to_record)
    valid_input = input_fn_builder(valid_files, False, _from_mention_to_record)
    return train_input, valid_input


def get_files(data_folder):
    if "gs://" in data_folder:
        from google.cloud import storage
        storage_client = storage.Client(project=project_id)
        bucket_name = data_folder.split("/")[2]
        bucket = storage_client.get_bucket(bucket_name)
        prefix = "/".join([x for x in data_folder.split("/")[3:]])
        files = [f"gs://{bucket_name}/{x.name}" for x in bucket.list_blobs(prefix=prefix)]
    else:
        # One folder -> dev, train/test split
        files = []
        for root, dirs, current_files in os.walk(data_folder):
            files += [f"{root}/{f}" for f in current_files if f.endswith(".tsv")]

    return files


def get_data(folder, batch_size, reduction, buffer_size, ignore_cumsum):
    train_input_fn, eval_input_fn = slice_and_split_input(folder)

    params = input_fn_params(batch_size, reduction, buffer_size, ignore_cumsum)
    train_data = train_input_fn(params)
    eval_data = eval_input_fn(params)
    return train_data, eval_data


def input_fn_params(batch_size, reduction, buffer_size, ignore_cumsum):
    return {"batch_size": batch_size,
            "reduction": reduction,
            "max_num_mentions": MAX_NUM_MENTIONS,
            "input_embeddings_size": INPUT_EMBEDDINGS_SIZE,
            "max_num_predictions": MAX_NUM_PREDICTIONS,
            "buffer_size": buffer_size,
            "ignore_cumsum": ignore_cumsum}


def train_eval(data_folder, extra_callbacks=[], return_model=False, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
               validation_steps=VALIDATION_STEPS, steps_per_epoch=(VALIDATION_STEPS * VALIDATIONS_PER_EPOCH),
               epochs=EPOCHS, reduction_steps=REDUCTIONS_STEPS):
    model, encoder = get_model()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, verbose=1),
                 tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1)
                 ] + extra_callbacks

    histories = []
    for reduction in np.linspace(1.01, 0., reduction_steps + 1)[1:]:
        K.set_value(model.optimizer.lr, 0.01)
        print(f"New Reduction {reduction}")
        train_data, eval_data = get_data(data_folder, batch_size, reduction, buffer_size, ignore_cumsum=False)

        history = model.fit(train_data,
                            epochs=epochs,
                            validation_data=eval_data,
                            validation_steps=validation_steps,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)
        histories.append(history)

    return [histories, encoder] + ([model] if return_model else [])


def cross_validate(data_folder, extra_callbacks=None, n_split=5):
    if extra_callbacks is None:
        extra_callbacks = []

    model, encoder = get_model()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_f1', mode='max', verbose=1),
                 tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='val_f1', mode='max', verbose=1)
                 ] + extra_callbacks

    files = get_files(data_folder)
    if len(files) == 0:
        print(f"No files found in folder {data_folder}")
        return

    histories = []
    for train_index, test_index in KFold(n_split).split(files):
        train_input = input_fn_builder([files[i] for i in train_index],
                                       True)
        valid_input = input_fn_builder([files[i] for i in test_index],
                                       False)
        train_data = train_input({"batch_size": BATCH_SIZE})
        eval_data = valid_input({"batch_size": BATCH_SIZE})

        history = model.fit(train_data,
                            epochs=EPOCHS,
                            validation_data=eval_data,
                            validation_steps=VALIDATION_STEPS,
                            steps_per_epoch=VALIDATION_STEPS * 10,
                            callbacks=callbacks)

        histories.append(history)

    return histories


if __name__ == "__main__":
    DATA_FOLDER = r"D:\ProjetoFinal\data\tsv\separation_40"
    MODELS_FOLDER = "D:\ProjetoFinal\model"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f'{MODELS_FOLDER}/model_ad.h5', verbose=1, save_best_only=True)
    ]
    history, encoder, model = train_eval(DATA_FOLDER, callbacks, True)
    encoder.save_weights(f"{MODELS_FOLDER}/encoder.h5")
    # cross_validate(r"D:\ProjetoFinal\data\debug")
