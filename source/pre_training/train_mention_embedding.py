import tensorflow as tf
import os
from bert import modeling
from bert.optimization import AdamWeightDecayOptimizer
from datetime import datetime
from sklearn.model_selection import train_test_split
from pre_training.model import BertModel

train_hooks = []
eval_hooks = []

BATCH_SIZE = 6
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 500
KEEP_CHECKPOINT_MAX = 2
LOG_N_ITER = 50
SAVE_SUMMARY_STEPS = 500
NUM_CPU_READ = 4
BUFFER_SIZE = 100
MAX_CLASSES = 2

MAX_NUM_MENTIONS = 20
MAX_NUM_PREDICTIONS = int(MAX_NUM_MENTIONS * (MAX_NUM_MENTIONS - 1) / 2)
INPUT_EMBEDDINGS_SIZE = 768

if "DEBUG" in os.environ:
    BATCH_SIZE = 2
    SAVE_CHECKPOINTS_STEPS = 30
    SAVE_SUMMARY_STEPS = 10
    KEEP_CHECKPOINT_MAX = 2
    LOG_N_ITER = 10

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, train_all=True):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    if train_all:
        tvars = tf.trainable_variables()
    else:
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bert/pooler")
        tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cls/predictions")
        tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bert/encoder/layer_11")

    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op, optimizer


def build_estimator(total_samples, num_epochs, model_dir, bert_config_file, batch_size=None, train_all=True):
    if batch_size is None:
        batch_size = BATCH_SIZE

    # Compute # train and warmup steps from batch size
    num_train_steps = int(total_samples * num_epochs / batch_size)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    # Create an input function for training. drop_remainder = True for using TPUs.

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        keep_checkpoint_max=KEEP_CHECKPOINT_MAX)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        train_all=train_all)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size)

    return estimator


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=NUM_CPU_READ):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_embeddings":
                tf.FixedLenFeature(max_seq_length * INPUT_EMBEDDINGS_SIZE, tf.float32),
            "input_mask":
                tf.FixedLenFeature(max_seq_length, tf.float32),
            "labels":
                tf.FixedLenFeature(max_predictions_per_seq, tf.float32),
            "output_mask":
                tf.FixedLenFeature(max_predictions_per_seq, tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=BUFFER_SIZE)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    out_seq = positions.shape[1]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return tf.reshape(output_tensor, (batch_size, out_seq, width))


def get_masked_lm_output(bert_config, input_tensor, label_ids, label_mask):
    """Get loss and log probs for the masked LM."""

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "custom_output_bias",
            shape=[MAX_CLASSES],
            initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, label_mask, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_mask = tf.reshape(label_mask, [-1])

        one_hot_labels = tf.reshape(tf.cast(label_ids, tf.float32), tf.shape(logits))

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_mask * tf.reshape(per_example_loss, [-1]))
        denominator = tf.reduce_sum(label_mask) + 1e-5
        loss = numerator / denominator

        # DEBUG
        # inspect = tf.arg_max(tf.reduce_sum(one_hot_labels, axis=-1), 0)
        # masked_lm_logits = tf.reshape(logits, [-1, logits.shape[-1]])
        # masked_lm_logits = tf.nn.sigmoid(masked_lm_logits)
        # masked_lm_predictions = tf.cast(tf.greater(masked_lm_logits, 0.5), tf.int64)

        # add = tf.add(tf.multiply(one_hot_labels, 128), 1)
        # shape = tf.shape(one_hot_labels)
        # broadcast = tf.transpose(tf.broadcast_to(tf.reshape(label_weights, [-1]), [shape[1], shape[0]]))
        # boosted_lm_weights = tf.multiply(tf.cast(add, tf.float32), broadcast)

        train_hooks.append(
            tf.estimator.LoggingTensorHook({
                # "label_ids": label_ids,
                # "sum_label_ids": tf.reduce_sum(label_ids),
                # "inspect": inspect,
                # "masked_lm_logits": masked_lm_logits[inspect],
                # "logits": logits[inspect],
                # "boosted_lm_weights": boosted_lm_weights[inspect],
                # "mult": mult[inspect],
                # "add": add[inspect],
                # "broadcast": broadcast[inspect],
                # "masked_lm_predictions": masked_lm_predictions[inspect],
                # "masked_lm_predictions shape": tf.shape(masked_lm_predictions),
                # "masked_lm_ids": masked_lm_ids[inspect],
                # "masked_lm_ids shape": tf.shape(masked_lm_ids),
                # "masked_lm_weights": masked_lm_weights[inspect],
                # "masked_lm_weights shape": tf.shape(masked_lm_weights),
                # "loss": loss,
                # "one_hot_labels shape": tf.shape(one_hot_labels),
                # "one_hot_labels": one_hot_labels[inspect],
                # "k_hot_count": tf.reduce_sum(one_hot_labels, axis=-1),
                # "max k_hot_count": tf.reduce_max(tf.reduce_sum(one_hot_labels, axis=-1)),
                # "per_example_loss shape": tf.shape(per_example_loss),
                # "per_example_loss": per_example_loss,
                # "label_weights shape": tf.shape(label_weights),
                # "label_weights": label_weights,
                # "log_probs shape": tf.shape(log_probs),
                # "log_probs": log_probs[inspect],
                # "masked_lm_log_probs": masked_lm_log_probs[inspect],
                # "denominator": denominator,
                # "numerator": numerator,
                # "max log_probs": tf.math.reduce_max(log_probs, axis=-1)[inspect],
                # "argmax log_probs": tf.argmax(log_probs, axis=-1, output_type=tf.int32)[inspect]
            }, every_n_iter=LOG_N_ITER))

        # train_hooks.clear()

    return loss, per_example_loss, logits


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(bert_config, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu=False,
                     use_one_hot_embeddings=False, train_all=True):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_embeddings = features["input_embeddings"]
        input_embeddings = tf.reshape(input_embeddings,
                                      shape=[input_embeddings.shape[0], MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE])
        input_mask = features["input_mask"]
        labels = features["labels"]
        output_mask = features["output_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_embeddings,
            input_mask=input_mask)

        (masked_lm_loss, masked_lm_example_loss, masked_lm_logits) = get_masked_lm_output(bert_config,
                                                                                          model.get_sequence_output(),
                                                                                          labels, output_mask)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, optimizer = create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                   use_tpu, train_all)
            train_hooks.append(tf.estimator.LoggingTensorHook({
                "lr": optimizer.learning_rate,
                "loss": total_loss
            }, every_n_iter=LOG_N_ITER))

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=train_hooks)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_logits, masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_logits,
                                                 [-1, masked_lm_logits.shape[-1]])

                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])

                # Any label is OK
                masked_lm_ids = tf.reshape(masked_lm_ids, masked_lm_log_probs.shape)
                masked_lm_ids = tf.argmax(tf.cast(masked_lm_ids, tf.float32) * masked_lm_log_probs, axis=-1)

                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])

                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)

                masked_lm_precision = tf.metrics.precision(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)

                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_precision": masked_lm_precision,
                    "masked_lm_loss": masked_lm_mean_loss
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_logits, labels, output_mask
            ])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn,
                evaluation_hooks=eval_hooks)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def slice_and_split_input(data_folder):
    if isinstance(data_folder, str):
        # One folder -> dev, train/test split
        files = os.listdir(data_folder)
        if len(files) == 0:
            print(f"No files found in folder {data_folder}")
            return

        files = [f"{data_folder}/{f}" for f in files if f.endswith(".tsv")]
        train_files, valid_files = train_test_split(files)
    else:
        # Two folders -> train/test already splitted
        train_files = os.listdir(data_folder[0])
        train_files = [f"{data_folder}/{f}" for f in train_files if f.endswith(".tsv")]

        valid_files = os.listdir(data_folder[1])
        valid_files = [f"{data_folder}/{f}" for f in valid_files if f.endswith(".tsv")]

    train_input = input_fn_builder(train_files,
                                   MAX_NUM_MENTIONS,
                                   MAX_NUM_PREDICTIONS,
                                   True)
    valid_input = input_fn_builder(valid_files,
                                   MAX_NUM_MENTIONS,
                                   MAX_NUM_PREDICTIONS,
                                   False)
    return train_input, valid_input


def train(data_dir, model_dir, bert_config_file, batch_size=BATCH_SIZE, total_samples=300000, num_train_epochs=10,
          train_all=True):
    print("Loading Dataset")
    estimator = build_estimator(total_samples, num_train_epochs, model_dir, bert_config_file, batch_size, train_all)

    current_time = datetime.now()
    print(f'Beginning Training! {num_train_epochs}')

    train_input_fn, eval_input_fn = slice_and_split_input(data_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    for epoch in range(num_train_epochs):
        print(f"New Epoch {epoch}")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        train_hooks.clear()
        eval_hooks.clear()

    print("Training took time ", datetime.now() - current_time)


if __name__ == "__main__":
    data_dir = r"D:\ProjetoFinal\data\debug"
    model_dir = r"D:\GDrive\Puc\Projeto Final\Datasets\mention_training"
    bert_file = r"D:\ProjetoFinal\model\debug_mention/mention_embedding_config.json"
    train(data_dir, model_dir, bert_file)
