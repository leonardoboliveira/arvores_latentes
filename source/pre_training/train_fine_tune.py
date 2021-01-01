from pre_training.fine_tune_bert import build_estimator, input_fn_builder, train_hooks, eval_hooks, BATCH_SIZE
from datetime import datetime
import tensorflow as tf
from pre_training.constants import BERT_MODEL_NAME, MAX_SEQ_LEN, MAX_NUM_PREDICTIONS
import os
from sklearn.model_selection import train_test_split


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
                                   MAX_SEQ_LEN,
                                   MAX_NUM_PREDICTIONS,
                                   True)
    valid_input = input_fn_builder(valid_files,
                                   MAX_SEQ_LEN,
                                   MAX_NUM_PREDICTIONS,
                                   False)
    return train_input, valid_input


def train(data_dir, model_dir, bert_dir, batch_size=BATCH_SIZE, total_samples=300000, num_train_epochs=10,
          train_all=True):
    print("Loading Dataset")
    estimator = build_estimator(total_samples, num_train_epochs, model_dir, bert_dir, batch_size, train_all)

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
    data_dir = r"D:\GDrive\Puc\Projeto Final\Datasets\finetuning\devel\positional\tf_full_seq_1"
    model_dir = r"d:\Users\loliveira\Documents\ProjetoFinal\model"
    bert_dir = f"D:/GDrive/Puc/Projeto Final/models/{BERT_MODEL_NAME}"
    train(data_dir, model_dir, model_dir, train_all=False)
