#!/bin/bash

#set -x

MAX_LINES=$1
echo Splitting in $MAX_LINES


cd data
for BASE_DATASET in devel
do
	rm -Rf $BASE_DATASET	
	for MODIFY in "$@" #conll tuned encoded B1 B2 B3 B4 B5 B6 F4 F5 F6 E2 E3 E4 M1 M1_ft
	do
		if [ "$MODIFY" = "NONE" ]; then
			echo "None modification. Skipping"
			continue
		fi

		FILENAME=$BASE_DATASET.$MODIFY
		if [ -f "$FILENAME" ]; then
			echo Processing $FILENAME
			mkdir -p $BASE_DATASET/$MODIFY
			
			python split_conll_file.py $FILENAME ./$FILENAME $BASE_DATASET/$MODIFY/ $MAX_LINES &
		fi
	done
	wait
done
