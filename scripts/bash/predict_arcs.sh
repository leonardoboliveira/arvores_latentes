#!/bin/bash

NAME=$1
IN_FILE=$2
export DATASET=devel

SCORER=${SCORER:-/home/loliveira/anaconda3/envs/cort/lib/python3.6/site-packages/cort/reference-coreference-scorers/v8.01/scorer.pl}
FLAGS_FILE=${FLAGS_FILE:-./flags.sh}

. $FLAGS_FILE

export PICKLED_PATH=$PICKLED_PATH.arc
export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
export PYTHONPATH=${PYTHONPATH:-../../source}

export TMPDIR=${TMPDIR:-/tmp}

python ../predicting/predict_arcs.py  -in $IN_FILE -out $NAME.predicted -ante $NAME.ante -clusterer extension.clusterer.all_ante -model $FULL_MODEL

perl $SCORER all $IN_FILE $NAME.predicted none | tee -a $NAME.metric
