#!/bin/bash

# Gerando encodings
python /code/data_handling/create_bert_embeddings.py /model/encoder ${MODEL_CHECKPOINT} /data/conll /data/encoded ${MODEL_TAG}

# Gerando bins
python /code/data_handling/create_bins_files.py /data/encoded /model/extra/deciles_${MODEL_TAG}.csv