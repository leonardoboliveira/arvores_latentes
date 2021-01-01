#!/bin/bash -e

set -x

DATASET=${DATASET:-devel}

MODEL_PATH=${MODEL_ROOT:-../../models/cort}
DATA_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET
export PYTHONPATH=${PYTHONPATH:-../../source}
SCORER=${SCORER:-/home/loliveira/anaconda3/envs/cort/lib/python3.6/site-packages/cort/reference-coreference-scorers/v8.01/scorer.pl}

#export PICKLE_PATH=${PICKLE_PATH:-$DATA_PATH/pickle}
#mkdir -p $PICKLE_PATH

if [[ "$DATASET" == "devel" ]]; then
  GOLD=$DATA_PATH/conll/$DATASET.conll.test.0
else
  GOLD=$DATA_PATH/conll/$DATASET.conll
fi


for FILE in "$MODEL_PATH"/*.7z
do
  WORK_DIR=$(mktemp -d)
  echo $FILE $WORK_DIR

  MODEL=$WORK_DIR/model.obj
  SPLIT=(${FILE//./ })
  OUT_PATTERN=$WORK_DIR/${SPLIT[-2]}

  7za x $FILE model.obj features.txt flags.sh -o$WORK_DIR -y

  if [[ -f "$WORK_DIR/flags.sh" ]]; then
    chmod 754 $WORK_DIR/flags.sh
    . $WORK_DIR/flags.sh
  fi

  export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE

  cort-predict-conll -in $GOLD \
      -model $MODEL \
      -out $WORK_DIR\predited \
      -extractor extension.antecedent_trees.extract_substructures_limited \
      -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
      -clusterer extension.clusterer.all_ante \
      -features $WORK_DIR/features.txt \
      -ante $OUT_PATTERN.ante \
      -instance_extractor extension.instance_extractors_mt.InstanceExtractor | tee $OUT_PATTERN.out

  perl $SCORER all $GOLD $WORK_DIR\predited none | tee -a $OUT_PATTERN.out

  7za a $FILE $OUT_PATTERN*

  rm -Rf $WORK_DIR
done

sudo shutdown now