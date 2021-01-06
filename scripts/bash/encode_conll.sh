#!/bin/bash

OUT_FOLDER=/data/$DATASET/${MODEL_TAG}
mkdir -p $OUT_FOLDER

# Gerando encodings
python /code/data_handling/create_bert_embeddings.py /model/encoder ${MODEL_CHECKPOINT} /data/original_conll $OUT_FOLDER ${MODEL_TAG}

# Gerando bins
python /code/data_handling/create_bins_files.py $OUT_FOLDER /model/extra/deciles_${MODEL_TAG}.csv