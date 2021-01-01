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

function run(){
	FILE_NAME=$1
	OUT_FILE=$(basename $1)
}

export -f run

find $INPUT_PATH -type f -exec basename {} \; | grep -v desktop | shuf | xargs -n 1 -P 3 -I '%' python create_spacy_embeddings.py $INPUT_PATH/'%' /mnt/rede/data/sliding/'%'.sliding
