#!/bin/bash -e

#set -x
MODEL=$1

echo Downloading Dataset
rclone copy Code:/Puc/Projeto\ Final/Datasets/tuned/devel.$MODEL.7z data/devel.$MODEL.7z

echo Splitting file
7za x data/devel.$MODEL.7z -odata/
./split.sh 30000 conll $MODEL
rm data/devel.$MODEL

echo Training model
./train.sh $MODEL

echo Evaluating model
./eval.sh $MODEL
