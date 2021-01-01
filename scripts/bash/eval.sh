#!/bin/bash -e

set -x

FILE=$1
GOLD_PATH=$2

DATASET=${DATASET:-devel}

MODEL_PATH=${MODEL_ROOT:-../../models/cort}
DATA_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET
export PYTHONPATH=${PYTHONPATH:-../../source}
SCORER=${SCORER:-/home/loliveira/anaconda3/envs/cort/lib/python3.6/site-packages/cort/reference-coreference-scorers/v8.01/scorer.pl}

export WORK_DIR=$(mktemp -d)
echo $FILE $WORK_DIR

export MODEL=$WORK_DIR/best.obj
SPLIT=(${FILE//./ })
OUT_PATTERN=$WORK_DIR/$DATASET.${SPLIT[-2]}

7za e $FILE -o$WORK_DIR -y


if [[ -f "$WORK_DIR/flags.sh" ]]; then
  chmod 754 $WORK_DIR/flags.sh
  export EXTRA_ROOT=$WORK_DIR
  . $WORK_DIR/flags.sh
fi

export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
echo "Using encoding from $BERT_DB_PATH"
mkdir $WORK_DIR/gold
mkdir $WORK_DIR/predict

run_predict(){
  GOLD=$1
  echo "Predicting $GOLD"
  TMP_PREDICT=$(mktemp --tmpdir=$WORK_DIR/predict)

  cp $GOLD  $WORK_DIR/gold/
  cort-predict-conll -in $GOLD \
      -model $MODEL \
      -out $TMP_PREDICT \
      -extractor extension.antecedent_trees.extract_substructures_limited \
      -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
      -clusterer extension.clusterer.all_ante \
      -features $WORK_DIR/features.txt \
      -instance_extractor extension.instance_extractors.InstanceExtractor
}

export -f run_predict

PARAMS=$(mktemp)

# find $GOLD_PATH -name *gold* > $PARAMS
find $GOLD_PATH -type f -name *conll > $PARAMS

parallel -j 12 run_predict < $PARAMS
#for PRM in $(find $GOLD_PATH -type f -name *conll)
#do
#    run_predict $PRM
#done

rm $PARAMS

echo "Finished predicting. Measuring"

cat $WORK_DIR/predict/* > $WORK_DIR/final_predict
cat $WORK_DIR/gold/* > $WORK_DIR/final_gold

perl $SCORER all $WORK_DIR/final_gold $WORK_DIR/final_predict none | tee -a $OUT_PATTERN.out

7za a $FILE $OUT_PATTERN*

# rm -Rf $WORK_DIR

# sudo shutdown now
