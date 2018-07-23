#!/bin/bash
python synth.py -c $1
python synth.py -c $1 --fun mean
python join.py -c $1 --suff projection mean

python synth.py -c $2 --centroids "data/chunk$1_centroid.json"
python synth.py -c $2 --fun mean
python join.py -c $2 --suff projection mean

python process.py -T "data/chunk$1_join.csv" -t "data/chunk$2_join.csv" -u

python forest.py -T $1 -t $2 --base join_select
