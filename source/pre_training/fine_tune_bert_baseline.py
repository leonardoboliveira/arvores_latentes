import tensorflow as tf

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from datetime import datetime
import tensorflow_hub as hub
import shutil
import sys
import pickle
import tf_metrics

hooks = []
debug = {}

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
# Compute train and warmup steps from batch size
BATCH_SIZE = 8
MAX_EXAMPLES = 2000
LEARNING_RATE = 2e-5
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
OUTPUT_DIR = "BERT"
MAX_SEQ_LEN = 512
num_classes = 768
pos_indices = list(range(1, num_classes))
average = 'micro'
features_path = sys.argv[-1]


# features_path = f"{root}/prepared/bert_finetuning3"

def build_estimator(labels):
    # Compute # train and warmup steps from batch size
    total_samples = len(labels)
    num_train_steps = int(total_samples / BATCH_SIZE)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    # Create an input function for training. drop_remainder = True for using TPUs.

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        num_labels=0,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    return estimator


def fix_y_df(df):
    df = df.apply(lambda row: [-(x + 1) if x <= 0 else x for x in row])
    # Moving all 1 to get all positive
    df = df + 1
    # assert sum(df.sum(axis=1) == 0) == 0, "No data"
    return df


def load_features(path, max_size=10000):
    labels = []
    tokens = []
    ids = []

    if os.path.isfile("all_features.dmp"):
        with open("all_features.dmp", "rb")as f:
            labels, tokens, ids = pickle.load(f)
    else:
        for file_name in tqdm(sorted(list(os.listdir(path)))):
            df = pd.read_csv(f"{path}/{file_name}", header=None, names=list(range(MAX_SEQ_LEN)), index_col=None)

            if file_name.endswith("y"):
                # Changing -1 and 0
                labels.append(fix_y_df(df))
            elif file_name.endswith("x1"):
                tokens.append(df)
            elif file_name.endswith("x2"):
                ids.append(df)

            if len(tokens) >= max_size and \
                    len(labels) >= max_size and \
                    len(ids) >= max_size:
                break

        with open("all_features.dmp", "wb") as f:
            pickle.dump((labels, tokens, ids), f)

    masks = [k.sum(axis=1).values > 0 for k in ids]
    for i in range(len(masks)):
        labels[i] = labels[i][masks[i]]
        tokens[i] = tokens[i][masks[i]]
        ids[i] = ids[i][masks[i]]

    labels = [labels[i] for i in range(len(masks)) if len(labels[i]) > 0]
    tokens = [tokens[i] for i in range(len(masks)) if len(tokens[i]) > 0]
    ids = [ids[i] for i in range(len(masks)) if len(ids[i]) > 0]

    return tokens, ids, labels


def slice_input(epoch, o_tokens, o_ids, o_labels, train=True):
    labels = []
    tokens = []
    ids = []

    def slice(df):
        full = len(df) / 5
        chunk = int(epoch % full)
        # Doing a 80-20 train-test split
        slice_start = chunk * 5
        slice_end = slice_start + 4

        if not train:
            slice_start = slice_end
            slice_end = slice_start + 1

        return df.iloc[slice_start:slice_end, :]

    for i in range(len(o_tokens)):
        labels.append(slice(o_labels[i]))
        tokens.append(slice(o_tokens[i]))
        ids.append(slice(o_ids[i]))

    def fix_it(df):
        print(f"list values len {len(df)}")
        print(f"shape 0 {df[0].shape}")
        df = pd.concat(df)
        print(f"New df:{df.shape}")
        return df.fillna(0).values.astype('int32')

    sliced_labels = fix_it(labels)
    sliced_tokens = fix_it(tokens)
    sliced_ids = fix_it(ids)
    sliced_mask = (sliced_ids > 0).astype('int32')

    return input_fn_builder(sliced_tokens, sliced_ids, sliced_labels, sliced_mask, train, False)


def input_fn_builder(tokens, ids, labels, mask, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    print(f"Tokens shape:{tokens.shape}")

    num_examples, seq_length = tokens.shape

    print(f"t {num_examples}, {seq_length}")

    all_input_ids = tokens
    all_input_mask = mask
    all_segment_ids = ids
    all_label_ids = labels

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": all_input_ids,
            "input_mask": all_input_mask,
            "segment_ids": all_segment_ids,
            "label_ids": all_label_ids
        })

        if is_training:
            # d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        d = d.prefetch(batch_size)
        return d

    return input_fn


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    print(f"input_ids:{input_ids.shape}")
    print(f"input_mask:{input_mask.shape}")
    print(f"segment_ids:{segment_ids.shape}")
    print(f"labels:{labels.shape}")

    print(f"bert_inputs:{bert_inputs.keys()}")
    print(f"bert_outputs:{bert_outputs.keys()}")

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    # output_layer = bert_outputs["sequence_output"]

    brick = np.zeros(num_classes)
    brick[1] = 1
    wall = np.reshape(np.tile(brick, MAX_SEQ_LEN), (MAX_SEQ_LEN, num_classes))
    wall = tf.convert_to_tensor(wall, dtype=tf.float32)

    output_layer = tf.broadcast_to(wall, tf.shape(bert_outputs["sequence_output"]))

    print(f"output_layer shape:{output_layer.shape}")
    print(f"pooled_output shape:{bert_outputs['pooled_output'].shape}")
    print(f"sequence_output shape:{bert_outputs['sequence_output'].shape}")

    hidden_size = output_layer.shape[-1].value
    # num_labels = input_ids.shape[1]
    one_hot_labels = tf.one_hot(labels, hidden_size, name="My_OneHot")
    print(f"one_hot_labels shape:{one_hot_labels.shape}")

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        # output_layer = tf.nn.dropout(output_layer, keep_prob=0.9, name="My_Dropout")
        # print(f"1 output_layer shape:{output_layer.shape}")

        # output_layer = tf.nn.softmax(output_layer, name="My Softmax")
        # print(f"2 output_layer shape:{output_layer.shape}")

        # tmp_mask = tf.broadcast_to(tf.expand_dims(input_mask, axis=-1), tf.shape(output_layer))
        # output_layer = tf.math.multiply(output_layer, tf.cast(tmp_mask, tf.float32), name="My_Argmax_Mask")

        softmax = tf.nn.softmax(output_layer, name="My_Softmax")
        print(f"2 softmax shape:{softmax.shape}")

        argmax = tf.math.argmax(softmax, axis=-1, name="My_Argmax")

        i = 6

        print(f"2 argmax shape:{argmax.shape}")
        predicted_labels = tf.math.multiply(argmax, tf.cast(input_mask, tf.int64), name="My_Argmax_Mask")

        hooks.append(
            tf.estimator.LoggingTensorHook(
                {"output_layer": output_layer[0],
                 "label": labels[0],
                 "predicted": predicted_labels[0],
                 "OK labels": tf.math.reduce_sum(
                     tf.to_float(tf.math.equal(tf.cast(labels, tf.int64), predicted_labels))),
                 "Total Labels": tf.shape(labels)},
                every_n_iter=10))

        if is_predicting:
            return predicted_labels

        print(f"logits : {output_layer.shape}")
        print(f"labels : {one_hot_labels.shape}")

        # output_layer = tf.reshape(output_layer, [BATCH_SIZE ,MAX_SEQ_LEN, 768])
        # print(f"logits : {output_layer.shape}")
        # print(f"tmp : {tmp.shape}")

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=output_layer, name="My_Loss")
        no_loss = tf.math.multiply(loss, tf.cast(input_mask, tf.float32), name="My_Multiply")
        rm = tf.reduce_mean(no_loss, name="My_ReduceMean")

        hooks.append(
            tf.estimator.LoggingTensorHook({
                "count no_loss": tf.math.count_nonzero(loss),
                "count input_mask": tf.math.count_nonzero(input_mask),
                "loss shape": tf.shape(loss),
                "rm shape": tf.shape(rm),
                "no_loss": no_loss[0],
                "loss": loss[0],
                "no_loss shape": tf.shape(no_loss),
                "one_hot_labels shape": tf.shape(one_hot_labels)
            }, every_n_iter=10))

        return rm, predicted_labels


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print(f"model_fn: input_ids:{input_ids.shape}")
        print(f"model_fn: input_mask:{input_mask.shape}")
        print(f"model_fn: segment_ids:{segment_ids.shape}")
        print(f"model_fn: label_ids:{label_ids.shape}")

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            hooks.append(tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10))

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                print(f"metric label_ids {label_ids.shape}")
                print(f"metric predicted_labels {predicted_labels.shape}")

                y_true, y_pred = label_ids, predicted_labels

                precision = tf_metrics.precision(
                    y_true, y_pred, num_classes, pos_indices, average=average)
                recall = tf_metrics.recall(
                    y_true, y_pred, num_classes, pos_indices, average=average)
                f2 = tf_metrics.fbeta(
                    y_true, y_pred, num_classes, pos_indices, average=average, beta=2)
                f1 = tf_metrics.f1(
                    y_true, y_pred, num_classes, pos_indices, average=average)

                return {
                    "precision": precision,
                    "recall": recall,
                    "f2": f2,
                    "f1": f1
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:

                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op,
                                                  training_hooks=hooks)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


if __name__ == "__main__":
    import os

    print(os.getcwd())

    tf.logging.set_verbosity(tf.logging.DEBUG)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    current_time = datetime.now()
    # "../../extra_files/bert_finetuning/"
    tokens, ids, labels = load_features(features_path, MAX_EXAMPLES)

    NUM_TRAIN_EPOCHS = int(max([len(x) for x in labels]) / 5)
    print(f'Beginning Training! {NUM_TRAIN_EPOCHS}')

    estimator = build_estimator(labels)

    for epoch in range(NUM_TRAIN_EPOCHS):
        print(f"New Epoch {epoch}")
        train_input_fn = slice_input(epoch, tokens, ids, labels, True)
        eval_input_fn = slice_input(epoch, tokens, ids, labels, False)

        print(f"hooks:{len(hooks)}")
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        # estimator.train(input_fn=slice_input(epoch, tokens, ids, labels, True), hooks=hooks)
        hooks.clear()

    print("Training took time ", datetime.now() - current_time)
