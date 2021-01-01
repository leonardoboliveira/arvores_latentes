import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import shutil
from training.train_mentions import MAX_NUM_MENTIONS, MAX_NUM_PREDICTIONS, INPUT_EMBEDDINGS_SIZE


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


name_to_features = {
    "input_embeddings":
        tf.io.FixedLenFeature(MAX_NUM_MENTIONS * INPUT_EMBEDDINGS_SIZE, tf.float32),
    "input_mask":
        tf.io.FixedLenFeature(MAX_NUM_MENTIONS, tf.int64),
    "labels":
        tf.io.FixedLenFeature(MAX_NUM_PREDICTIONS, tf.int64),
    "output_mask":
        tf.io.FixedLenFeature(MAX_NUM_PREDICTIONS, tf.int64),
    "cumsum":
        tf.io.FixedLenFeature(MAX_NUM_PREDICTIONS * INPUT_EMBEDDINGS_SIZE * 5, tf.float32),
}


def to_feature(tensor):
    as_numpy = tensor.numpy()
    if as_numpy.dtype == np.float32:
        return create_float_feature(as_numpy)
    return create_int_feature(as_numpy)


def check(out_dir, ds):
    counter = 0
    for batch in tqdm(ds):
        new_file = f"{out_dir}/{counter}.tsv"
        print(f"New file:{new_file}")
        counter += 1
        with tf.io.TFRecordWriter(new_file) as writer:
            for raw_record in batch:
                as_dict = tf.io.parse_single_example(raw_record, name_to_features)
                features = {x: to_feature(as_dict[x]) for x in as_dict}
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())


main_root = r"C:\Users\loliveira\Desktop\teste"
work_dir = r"C:\Users\loliveira\Desktop\work"

os.makedirs(f"{work_dir}/data_in", exist_ok=True)
os.makedirs(f"{work_dir}/data_out", exist_ok=True)
for root, dirs, current_files in os.walk(main_root):
    for file_name in current_files:
        print(f"\n{file_name}")
        if "_" not in file_name:
            print("Ignoring")
            continue
        shutil.move(f"{root}/{file_name}", f"{work_dir}/data_in")
        ds = tf.data.TFRecordDataset(filenames=[f"{work_dir}/data_in/{file_name}"])
        ds = ds.batch(20)
        check(f"{work_dir}/data_out", ds)

        out_dir = os.path.join(root, file_name.replace(".tsv", ""))
        os.makedirs(out_dir, exist_ok=True)

        for src in os.listdir(f"{work_dir}/data_out"):
            shutil.move(f"{work_dir}/data_out/{src}", out_dir)
        os.remove(f"{work_dir}/data_in/{file_name}")
