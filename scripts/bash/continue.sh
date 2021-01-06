#!/bin/bash
export START_FROM=4
export MODEL=/tmp/tmp.EEBPu17JHm/model.obj
export FLAGS_FILE=/code/scripts/flags.sh
export FIRST_EPOCH=0
export BEST_CONLL=0
export TMPDIR=/tmp/tmp.EEBPu17JHm
export PICKLED_PATH=/data/train/pickle/span_40_induction
./train_eval.sh simples

