#!/bin/bash
for MODEL in $@
do
	./cv.sh $MODEL
done
