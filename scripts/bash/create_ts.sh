#!/bin/bash -e

INPUT_PATH=$1

function solve_file() {
  set -x

  IN_FILE=$1
  FLAGS_FILE=${FLAGS_FILE:-./flags.sh}
  EXTRA_FILES_PATH=${EXTRA_ROOT:-../../extra_files}

  . $FLAGS_FILE

  BERT_ENCODE=${BERT_ENCODE:-encoded}
  export BERT_DB_PATH=${DATA_ROOT:-/mnt/d/ProjetoFinal/data}/$DATASET/$BERT_ENCODE
  export PYTHONPATH=${PYTHONPATH:-../../source}
  export TMPDIR=${TMPDIR:-/tmp}

  OUTPUT_PATH=/mnt/rede/data/cumsum_$MAX_DISTANCE
  G_PATH=/ProjetoFinal/tsv/cumsum_$MAX_DISTANCE
  mkdir -p $OUTPUT_PATH

  echo $IN_FILE

  TMP_RCLONE=$(mktemp)
  rclone lsf Kadima:$G_PATH/ > $TMP_RCLONE
  COUNT_MISSING=$(cat $IN_FILE  | grep "#begin" | sed 's/#begin document //g' | sed 's/; part /_/g' | sed 's/(//g' | sed 's/)//g' | sed 's/\//_/g' | xargs -I {} bash -c "grep {} $TMP_RCLONE | wc -l" | grep 0 | wc -l)
  rm $TMP_RCLONE

  if [ $COUNT_MISSING -gt 0 ]; then
    echo "Need processing"
    cd ../data_handling
    OUTPUT_PATH=$(mktemp -d)
    python create_mention_embedding_ts.py $IN_FILE $OUTPUT_PATH
    cd $OUTPUT_PATH
    find . -type f | xargs -I {} basename {} | xargs -I {} rclone moveto {} Kadima:$G_PATH/{} --no-traverse --update --progress --ignore-existing
    rm -Rf $OUTPUT_PATH
  else
    echo "Skipping. All precessed"
  fi
}

export -f solve_file

find $INPUT_PATH -type f | shuf | xargs -n 1 -P 2 -I '{}' bash -c 'solve_file "$@"' _ {}
