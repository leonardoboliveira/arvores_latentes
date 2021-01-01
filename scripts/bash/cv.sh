#!/bin/bash -e

#set -x
MODEL=$1
SPLIT_SIZE=25000

#export SKIP_FEATURE_INDUCTION=1

if [ "$MODEL" != "NONE" ]; then
	echo Downloading Dataset
	rclone copy Code:/Puc/Projeto\ Final/Datasets/tuned/devel.$MODEL.7z data/devel.$MODEL.7z

	echo Splitting file
	7za x data/devel.$MODEL.7z -odata/
	./split.sh $SPLIT_SIZE conll $MODEL
	
	rm data/devel.$MODEL
else
	./split.sh $SPLIT_SIZE conll NONE
fi


count=0
total=$(ls data/devel/conll/ | wc -l)
((total-=1))

entries=($(shuf -i 0-$total -n 5))

for FOLD in "${entries[@]}"; do
	echo Training model $FOLD
	./train.sh $MODEL $FOLD

	echo Evaluating model $FOLD
	./eval.sh $MODEL $FOLD
done
