#!/bin/bash  -e

set -o xtrace

### Use external or default options
# May be defined
# PYTHONPATH
# DATA_ROOT
# EXTRA_ROOT
# OUT_ROOT
# MODEL_ROOT

DATASET=${DATASET:-devel}
PARTITIONS=${PARTITIONS:-11}
START_FROM=${START_FROM:-0}
FIRST_EPOCH=${FIRST_EPOCH:-0}

EXTRA_FILES_PATH=${EXTRA_ROOT:-../../extra_files}
DATA_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET
OUT_PATH=${OUT_ROOT:-/mnt/d/ProjetoFinal/output}
SCORER=${SCORER:-/home/loliveira/anaconda3/envs/cort/lib/python3.6/site-packages/cort/reference-coreference-scorers/v8.01/scorer.pl}
FLAGS_FILE=${FLAGS_FILE:-./flags.sh}
CONTINUE_FILE=${CONTINUE_FILE:-continue.sh}
#export PICKLE_PATH=${PICKLE_PATH:-$DATA_PATH/pickle}
#mkdir -p $PICKLE_PATH

. $FLAGS_FILE

EXPERIMENT=${EXPERIMENT:-noname}

BERT_ENCODE=${BERT_ENCODE:-encoded}
FINAL_MODEL=${MODEL_ROOT:-../../../models/cort}/$BERT_ENCODE.$DATASET.$MAX_DISTANCE.$EXPERIMENT.7z

COST_SCALING=${COST_SCALING:-1}

# export EXTRACT_SINGLE_THREAD=1
export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
export PYTHONPATH=${PYTHONPATH:-../../source}

export TMPDIR=${TMPDIR:-/tmp}
export TMPDIR=$(mktemp -d)
MODEL=${MODEL:-$TMPDIR/model.obj}

export PREDICTED=${PREDICTED:-$TMPDIR/predicted}
cp "$FLAGS_FILE" $TMPDIR
cp "$BINS_FILE" $TMPDIR
cp "$EXTRA_FILES_PATH/features.txt" $TMPDIR

rm -f FINAL_MODEL

BEST_CONLL=${BEST_CONLL:-0}

for EPOCH in $(seq $FIRST_EPOCH 5); do
for BATCH in $(seq $START_FROM $PARTITIONS); do
  echo Epoch $BATCH
  IN_FILE=$DATA_PATH/conll/$DATASET.conll.train.$BATCH

  echo Training using $IN_FILE | tee $OUT_PATH/$EXPERIMENT.out

  echo "#!/bin/bash
export START_FROM=$BATCH
export MODEL=$MODEL
export FLAGS_FILE=$FLAGS_FILE
export FIRST_EPOCH=$EPOCH
export BEST_CONLL=$BEST_CONLL
export TMPDIR=$TMPDIR
export PICKLED_PATH=$PICKLED_PATH
./train_eval.sh $EXPERIMENT
" > $CONTINUE_FILE

  cort-train -features $TMPDIR/features.txt \
    -in $IN_FILE \
    -out $MODEL \
    -extractor extension.antecedent_trees.extract_substructures_limited \
    -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
    -cost_function cort.coreference.cost_functions.cost_based_on_consistency \
    -instance_extractor extension.instance_extractors.InstanceExtractor \
    -model $MODEL \
    -n_iter 50 \
    -cost_scaling $COST_SCALING

  if ! ((($BATCH + 1) % 6)); then
    OUT_PATTERN=$OUT_PATH/$EXPERIMENT.$EPOCH.$BATCH
    GOLD=$DATA_PATH/conll/$DATASET.conll.test.0
    PREDICTED=$OUT_PATTERN.predicted

    echo Evaluating $GOLD
    cort-predict-conll -in $GOLD \
      -model $MODEL \
      -out $PREDICTED \
      -extractor extension.antecedent_trees.extract_substructures_limited \
      -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
      -clusterer cort.coreference.clusterer.all_ante \
      -features "$EXTRA_FILES_PATH/features.txt" \
      -ante $OUT_PATTERN.ante \
      -instance_extractor extension.instance_extractors.InstanceExtractor | tee $OUT_PATTERN.out

    perl $SCORER all $GOLD $PREDICTED none | tee -a $OUT_PATTERN.out
    CONLL=$(python "$PYTHONPATH/evaluations/calc_conll.py" $OUT_PATTERN.out)
    if (( $(echo "$CONLL > $BEST_CONLL" |bc -l) )); then
        echo "New Best CONLL:$CONLL" | tee -a $OUT_PATTERN.out
        cp $MODEL $TMPDIR/best.obj
        BEST_CONLL=$CONLL
    fi
  fi
done
START_FROM=0
done

echo Packing results
7za a $FINAL_MODEL $MODEL $EXTRA_FILES_PATH/* $OUT_PATH/$EXPERIMENT.* $TMPDIR/flags.sh $TMPDIR/best.obj -spf2
7za rn $FINAL_MODEL $(basename $MODEL) model.obj

rm $CONTINUE_FILE

if [ "$DO_NOT_SHUTDOWN" != "1" ]; then
	sudo shutdown now
fi
