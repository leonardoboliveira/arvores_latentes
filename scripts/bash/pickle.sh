#!/bin/bash

set -x

FLAGS_FILE=${FLAGS_FILE:-./flags.sh}
EXTRA_FILES_PATH=${EXTRA_ROOT:-../../extra_files}

. $FLAGS_FILE

BERT_ENCODE=${BERT_ENCODE:-encoded}
export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
export PYTHONPATH=${PYTHONPATH:-../../source}
export TMPDIR=${TMPDIR:-/tmp}

INPUT_PATH=$1
OUTPUT_PATH=$PICKLED_PATH

cd ../data_handling

find $INPUT_PATH -type f | shuf | xargs -n 1 -P 2 -I '{}' python pickle_arc_information.py $OUTPUT_PATH $MODEL_ROOT/encoder.h5 '{}'
