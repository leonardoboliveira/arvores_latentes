#!/bin/bash
export START_FROM=0
export MODEL=/mnt/rede/tmp/tmp.A7wEGjDMfS/model.obj
export FLAGS_FILE=./flags.sh
export FIRST_EPOCH=0
export BEST_CONLL=0
export TMPDIR=/mnt/rede/tmp/tmp.A7wEGjDMfS
export PICKLED_PATH=/mnt/rede/data/pickle/span_40_induction
./train_eval.sh span_induce_E6

