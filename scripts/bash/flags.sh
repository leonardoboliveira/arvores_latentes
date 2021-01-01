export SKIP_FEATURE_INDUCTION=0
export BERT_ENCODE=span
export MAX_DISTANCE=40
export PICKLED_PATH=${PICKLE_ROOT:-/mnt/rede/data}/pickle/${BERT_ENCODE}_${MAX_DISTANCE}
export ENCODER_MODEL=$MODEL_ROOT/encoder_${MAX_DISTANCE}_diagonal.h5
export FULL_MODEL=$MODEL_ROOT/model_${MAX_DISTANCE}_diagonal.h5
export INDUCED_PICKLEwqD=${EXTRA_ROOT}/features_${BERT_ENCODE}.dat
export COST_SCALING=20000

export BINS_FILE=${EXTRA_ROOT:-../../extra_files}/deciles_${BERT_ENCODE}.csv

if [[ "$SKIP_FEATURE_INDUCTION" == "0" ]]; then
    unset SKIP_FEATURE_INDUCTION
    export PICKLED_PATH=${PICKLED_PATH}_induction
elif [[ "$SKIP_FEATURE_INDUCTION" == "EFI" ]]; then
    export PICKLED_PATH=${PICKLED_PATH}_efi
fi
