import os
import tensorflow as tf
import tensorflow.keras.backend as K
from extension.bert_features import CUMSUM_SPLITS

MAX_NUM_MENTIONS = int(os.environ["MAX_DISTANCE"]) if "MAX_DISTANCE" in os.environ else 40
MAX_NUM_PREDICTIONS = int(MAX_NUM_MENTIONS * (MAX_NUM_MENTIONS - 1) / 2)
INPUT_EMBEDDINGS_SIZE = 768
OUTPUT_EMBEDDINGS_SIZE = 128


def dilated_cumsum(embeddings, output_mask, cumsum):
    h = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE, 1), name="Emb_Encoder")(embeddings)
    h = tf.keras.layers.Conv2D(256, (1, INPUT_EMBEDDINGS_SIZE), strides=(1, 1), activation='relu', name="Emb_Conv_1")(h)
    h = tf.keras.layers.BatchNormalization(name="Emb_BatchNorm_1")(h)
    h = tf.keras.layers.Permute((1, 3, 2), name="Emb_Permute_1")(h)
    h = tf.keras.layers.Conv2D(64, (1, 256), strides=(1, 1), activation='relu', name="Emb_Conv_2")(h)
    h = tf.keras.layers.BatchNormalization(name="Emb_BatchNorm_2")(h)
    h = tf.keras.layers.Permute((1, 3, 2), name="Emb_Permute_2")(h)

    dilated = []
    for i in range(1, MAX_NUM_MENTIONS):
        d = tf.keras.layers.Conv2D(OUTPUT_EMBEDDINGS_SIZE, (2, 64), strides=(1, 1), activation='relu',
                                   dilation_rate=(i, 1),
                                   name=f"Emb_Dilated_{i}", padding='valid')(h)
        dilated.append(d)

    h = tf.keras.layers.Concatenate(axis=1)(dilated)
    h = tf.keras.layers.BatchNormalization(name="Emb_BatchNorm_3")(h)
    print(f"Emb_BatchNorm_1.shape: {h.shape}")

    # Encoding between arcs
    print(f"cumsum:{cumsum.shape}")
    h2 = tf.keras.layers.Permute((1, 3, 2), name="Cumsum_Permute_1")(cumsum)
    print(f"Cumsum_Permute_1:{h2.shape}")
    h2 = tf.keras.layers.Conv2D(256, (1, INPUT_EMBEDDINGS_SIZE), strides=(1, 1), activation='relu',
                                name="Cumsum_Conv_1")(h2)
    h2 = tf.keras.layers.BatchNormalization(name="CumSum_BatchNorm_1")(h2)
    print(f"Cumsum_Conv_1:{h2.shape}")
    h2 = tf.keras.layers.Permute((1, 3, 2), name="Cumsum_Permute_2")(h2)
    print(f"Cumsum_Permute_2:{h2.shape}")
    h2 = tf.keras.layers.Conv2D(128, (1, 256), strides=(1, 1), activation='relu', name="Cumsum_Conv_2")(h2)
    print(f"Cumsum_Conv_2:{h2.shape}")
    h2 = tf.keras.layers.BatchNormalization(name="CumSum_BatchNorm_2")(h2)

    h = tf.keras.layers.Concatenate(axis=2, name="Final_Encoder")([h, h2])
    print(f"Final_Encoder.shape: {h.shape}")
    h = tf.keras.layers.Conv2D(MAX_NUM_PREDICTIONS, (2, 1), strides=(1, 1), activation='relu',
                               data_format='channels_first', name="Final_Conv_1")(h)
    print(f"Final_Conv_1.shape: {h.shape}")

    h = K.squeeze(h, -2)
    h = tf.keras.layers.BatchNormalization(name="Final_BatchNorm")(h)
    print(f"Final_BatchNorm.shape: {h.shape}")

    e = h

    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu', name="Decoder")(h)
    print(f"Decoder.shape: {h.shape}")
    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h

    return o, e


def dilated_simple(embeddings, output_mask):
    # i = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE, 1), name="ConvEncoder")(embeddings)

    h = embeddings
    print(f"1:{h.shape}")
    h = tf.keras.layers.Conv1D(MAX_NUM_MENTIONS, 3, strides=3, activation='relu', data_format='channels_first')(h)
    print(f"2:{h.shape}")
    h = tf.keras.layers.Conv1D(MAX_NUM_MENTIONS, 4, strides=4, activation='relu', data_format='channels_first')(h)
    print(f"3:{h.shape}")
    h = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, 64, 1))(h)
    print(f"4:{h.shape}")

    dilated = []
    for i in range(1, MAX_NUM_MENTIONS):
        d = tf.keras.layers.Conv2D(OUTPUT_EMBEDDINGS_SIZE, (2, 64), strides=(1, 1), activation='relu',
                                   dilation_rate=(i, 1),
                                   name=f"Dilated{i}", padding='valid')(h)
        dilated.append(d)

    h = tf.keras.layers.Concatenate(axis=1)(dilated)

    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Reshape((MAX_NUM_PREDICTIONS, OUTPUT_EMBEDDINGS_SIZE))(h)
    e = h

    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu', name="Decoder")(h)
    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h
    return o, e


def dilated(embeddings, output_mask):
    i = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE, 1), name="ConvEncoder")(embeddings)

    h = i
    # print(f"1:{h.shape}")
    h = tf.keras.layers.Conv2D(256, (1, INPUT_EMBEDDINGS_SIZE), strides=(1, 1), activation='relu')(h)
    # print(f"2:{h.shape}")
    h = tf.keras.layers.Permute((1, 3, 2))(h)
    # print(f"3:{h.shape}")
    h = tf.keras.layers.Conv2D(64, (1, 256), strides=(1, 1), activation='relu')(h)
    # print(f"4:{h.shape}")
    h = tf.keras.layers.Permute((1, 3, 2))(h)
    # print(f"5:{h.shape}")

    dilated = []
    for i in range(1, MAX_NUM_MENTIONS):
        d = tf.keras.layers.Conv2D(OUTPUT_EMBEDDINGS_SIZE, (2, 64), strides=(1, 1), activation='relu',
                                   dilation_rate=(i, 1),
                                   name=f"Dilated{i}", padding='valid')(h)
        dilated.append(d)

    h = tf.keras.layers.Concatenate(axis=1)(dilated)

    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Reshape((MAX_NUM_PREDICTIONS, OUTPUT_EMBEDDINGS_SIZE))(h)
    e = h

    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu', name="Decoder")(h)
    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h
    return o, e


def conv_encoder_smaller(embeddings, output_mask):
    # no dropout: 0.10622842609882355

    h = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE, 1), name="ConvEncoder")(embeddings)
    h = tf.keras.layers.Conv2D(256, (1, 768), strides=(1, 1), activation='relu')(h)
    h = tf.keras.layers.Permute((1, 3, 2))(h)
    h = tf.keras.layers.Conv2D(64, (1, 256), strides=(1, 1), activation='relu')(h)
    h = tf.keras.layers.Permute((1, 3, 2))(h)
    h = tf.keras.layers.Conv2D(16, (1, 64), strides=(1, 1), activation='relu')(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(MAX_NUM_PREDICTIONS * OUTPUT_EMBEDDINGS_SIZE, activation='relu')(h)

    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Reshape((MAX_NUM_PREDICTIONS, OUTPUT_EMBEDDINGS_SIZE))(h)
    e = h

    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu', name="Decoder")(h)
    # h = tf.keras.layers.Embedding(OUTPUT_EMBEDDINGS_SIZE, 2, input_length=MAX_NUM_PREDICTIONS)(h)
    # h = tf.keras.layers.Dense(MAX_NUM_PREDICTIONS * OUTPUT_EMBEDDINGS_SIZE, activation='relu')(h)
    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h
    return o, e


def conv_encoder(embeddings, output_mask):
    # no dropout: 0.10622842609882355

    h = tf.keras.layers.Reshape((MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE, 1), name="ConvEncoder")(embeddings)

    h = tf.keras.layers.Conv2D(128, (1, 768), strides=(1, 1), activation='relu')(h)

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(MAX_NUM_PREDICTIONS * OUTPUT_EMBEDDINGS_SIZE, activation='relu')(h)

    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Reshape((MAX_NUM_PREDICTIONS, OUTPUT_EMBEDDINGS_SIZE))(h)
    e = h

    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu')(h)
    # h = tf.keras.layers.Embedding(OUTPUT_EMBEDDINGS_SIZE, 2, input_length=MAX_NUM_PREDICTIONS)(h)
    # h = tf.keras.layers.Dense(MAX_NUM_PREDICTIONS * OUTPUT_EMBEDDINGS_SIZE, activation='relu')(h)
    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h
    return o, e


def one_layer(embeddings, output_mask):
    h = tf.keras.layers.Flatten()(embeddings)
    h = tf.keras.layers.Dense(MAX_NUM_PREDICTIONS * OUTPUT_EMBEDDINGS_SIZE, activation='relu')(h)
    h = tf.keras.layers.Reshape((MAX_NUM_PREDICTIONS, OUTPUT_EMBEDDINGS_SIZE))(h)
    e = h
    h = tf.keras.layers.Conv1D(2, 1, strides=1, activation='relu')(h)
    h = tf.keras.layers.Dropout(0.2)(h)
    # h = tf.keras.layers.Activation('softmax')(h)

    h = tf.keras.layers.Multiply()([h, output_mask])
    o = h
    return o, e


def recall(y_true, y_pred):
    # Count positive samples.
    y_true = y_true[:, :, 1]
    y_pred = y_pred[:, :, 1]

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1))) + 1e-5

    return c1 / c3


@tf.function
def precision(y_true, y_pred):
    # Count positive samples.
    y_true = y_true[:, :, 1]
    y_pred = y_pred[:, :, 1]

    x = K.sum(K.round(K.clip(y_true, 0, 1)))
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) + 1e-5

    # tf.print("Precision:",c1,"/",c2, "(",x,")")

    return c1 / c2


def f1_score(y_true, y_pred):
    # Count positive samples.
    y_true = y_true[:, :, 1]
    y_pred = y_pred[:, :, 1]

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) + 1e-5
    c3 = K.sum(K.round(K.clip(y_true, 0, 1))) + 1e-5

    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    return f1_score


def get_model():
    def create_model():
        output_mask = tf.keras.layers.Input(shape=(MAX_NUM_PREDICTIONS, 1), name="output_mask")
        embeddings = tf.keras.layers.Input(shape=(MAX_NUM_MENTIONS, INPUT_EMBEDDINGS_SIZE,), name="input_embeddings")
        cumsum = tf.keras.layers.Input(shape=(MAX_NUM_PREDICTIONS, CUMSUM_SPLITS, INPUT_EMBEDDINGS_SIZE), name="cumsum")

        output, encoder = dilated_cumsum(embeddings, output_mask, cumsum)

        model = tf.keras.Model([embeddings, output_mask, cumsum], output, name="complete")
        model.summary()

        encoder = tf.keras.Model([embeddings, cumsum], encoder, name="encoder")
        encoder.summary()

        # @tf.function
        def loss(y_true, y_pred):
            log_probs = tf.nn.log_softmax(y_pred, axis=-1)

            label_weights = output_mask
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = y_true

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            per_example_loss = tf.reshape(per_example_loss, [-1])

            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

            return loss

        return model, encoder, loss

    model, encoder, loss = create_model()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[precision, recall, f1_score])
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        print('Found TPU at: {}'.format(TPU_ADDRESS))
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))
        return tpu_model, encoder
    except KeyError:
        return model, encoder
