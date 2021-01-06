# Nome para a rodada. Soh para distinguir as diversas execucoes
export EXPERIMENT=simples

# Valores possiveis:
## - 0: vai usar inducao tipo SURFACE
## - 1: Nao sera feita inducao de features
## - EFI: Sera feita inducao tipo EFI
export SKIP_FEATURE_INDUCTION=0

# Nome do encoding que sera utilizado.
export BERT_ENCODE=${MODEL_TAG:-span}

# Distancia maxima entre duas mencoes. Acima disso nao serao consideradas com arcos canidatos
export MAX_DISTANCE=40

# Pasta onde serao salvos os arcos codificados
export PICKLED_PATH=${PICKLE_ROOT:-/mnt/rede/data}/pickle/${BERT_ENCODE}_${MAX_DISTANCE}

# Margem
export COST_SCALING=20000

# Arquivo de bins
export BINS_FILE=${EXTRA_ROOT:-../../extra_files}/deciles_${BERT_ENCODE}.csv

# Apenas para o modo de treino sem arvores latentes
export ENCODER_MODEL=$MODEL_ROOT/encoder_${MAX_DISTANCE}_diagonal.h5
export FULL_MODEL=$MODEL_ROOT/model_${MAX_DISTANCE}_diagonal.h5

if [[ "$SKIP_FEATURE_INDUCTION" == "0" ]]; then
  unset SKIP_FEATURE_INDUCTION
  export PICKLED_PATH=${PICKLED_PATH}_induction
elif [[ "$SKIP_FEATURE_INDUCTION" == "EFI" ]]; then
  export PICKLED_PATH=${PICKLED_PATH}_efi
fi
