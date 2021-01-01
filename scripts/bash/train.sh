#!/bin/bash  -e

#set -o xtrace

export PYTHONPATH=.
export TMPDIR=/home/tmp
rm -f $TMPDIR/*
rm -f tmp*

#source venv/bin/activate
#conda activate train

DATASET=devel
BERT_DB=$1
MODEL=models/model.$DATASET.$BERT_DB.obj
DATA_PATH=data/$DATASET
FOLD=${2:-0}

rm -f $MODEL

for f in $(ls $DATA_PATH/conll)
do
        echo Training using $f
	
	if [[ "$f" =~ \.$FOLD$ ]]; then
		echo is fold
		continue
	fi

        export BERT_DB_PATH=$DATA_PATH/$BERT_DB/${f/conll/$BERT_DB}

        cort-train -features extra/features.txt \
                   -in $DATA_PATH/conll/$f \
                   -out $MODEL \
                   -extractor extension.antecedent_trees.extract_substructures_limited \
                   -perceptron extension.antecedent_trees.AntecedentTreePerceptron \
                   -cost_function cort.coreference.cost_functions.cost_based_on_consistency \
                   -instance_extractor extension.instance_extractors.InstanceExtractor \
                   -model $MODEL \
		   -n_iter 5
done

#sudo shutdown now
