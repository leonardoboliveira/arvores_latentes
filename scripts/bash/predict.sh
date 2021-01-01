#!/bin/bash  -e

set -o xtrace
EXPERIMENT=${1:-noname}

### Use external or default options
# May be defined
# PYTHONPATH
# DATA_ROOT
# EXTRA_ROOT
# OUT_ROOT
# MODEL_ROOT

DATASET=devel
PARTITIONS=${PARTITIONS:-11}
START_FROM=${START_FROM:-0}
FIRST_EPOCH=${FIRST_EPOCH:-0}

export EXTRA_ROOT=$2
EXTRA_FILES_PATH=$EXTRA_ROOT
DATA_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET
OUT_PATH=$EXTRA_ROOT
SCORER=${SCORER:-/home/loliveira/anaconda3/envs/cort/lib/python3.6/site-packages/cort/reference-coreference-scorers/v8.01/scorer.pl}
FLAGS_FILE=${EXTRA_ROOT}/flags.sh

#export PICKLE_PATH=${PICKLE_PATH:-$DATA_PATH/pickle}
#mkdir -p $PICKLE_PATH

. $FLAGS_FILE

BERT_ENCODE=${BERT_ENCODE:-encoded}
FINAL_MODEL=${MODEL_ROOT:-../../../models/cort}/$BERT_ENCODE.$DATASET.$MAX_DISTANCE.$EXPERIMENT.7z

# export EXTRACT_SINGLE_THREAD=1
export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
export PYTHONPATH=${PYTHONPATH:-../../source}

export TMPDIR=${TMPDIR:-/tmp}
export TMPDIR=$(mktemp -d)
MODEL=${EXTRA_ROOT}/$3

IN_FOLDER=$DATA_ROOT/conll/development
OUT_PATTERN=$OUT_PATH/$EXPERIMENT
PREDICTED=$OUT_PATTERN.predicted
ANTE=$OUT_PATTERN.ante
echo > $PREDICTED

for GOLD in $(find $IN_FOLDER/data -type f | grep gold)
do
    TMP_PREDICTED=$(mktemp)
    TMP_ANTE=$(mktemp)
    echo Evaluating $GOLD

    cort-predict-conll -in $GOLD \
      -model $MODEL \
      -out $TMP_PREDICTED \
      -extractor extension.antecedent_trees.extract_substructures_limited \
      -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
      -clusterer cort.coreference.clusterer.all_ante \
      -features $EXTRA_FILES_PATH/features.txt \
      -ante $TMP_ANTE \
      -instance_extractor extension.instance_extractors.InstanceExtractor

    cat $TMP_PREDICTED >> $PREDICTED
    cat $TMP_ANTE >> $ANTE
    rm $TMP_PREDICTED
    rm $TMP_ANTE
done
perl $SCORER all $IN_FOLDER/devel.conll $PREDICTED none | tee $OUT_PATTERN.metrics
