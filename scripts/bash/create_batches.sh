#!/bin/bash

OUT_FOLDER=/data/${DATASET}/conll
mkdir -p $OUT_FOLDER
python /code/data_handling/split/split_train_test_conll.py /data/original_conll $OUT_FOLDER $DATASET.conll ${PARTITIONS} True