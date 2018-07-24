# #!/bin/bash

# PROJECTION
python synth.py -c $1
python synth.py -c $2 --centroids "data/chunk$1_centroid.json"

python synth.py -c $1 --metric dot
python synth.py -c $2 --metric dot --centroids "data/chunk$1_centroid.json"

python synth.py -c $1 --metric angle
python synth.py -c $2 --metric angle --centroids "data/chunk$1_centroid.json"

# CLASS PROJECTION
python synth.py -c $1 --fun class_projection
python synth.py -c $2 --fun class_projection --centroids "data/chunk$1_centroid.json"

python synth.py -c $1 --fun class_projection --metric dot
python synth.py -c $2 --fun class_projection --metric dot --centroids "data/chunk$1_centroid.json"

python synth.py -c $1 --fun class_projection --metric angle
python synth.py -c $2 --fun class_projection --metric angle --centroids "data/chunk$1_centroid.json"

# python synth.py -c $1 --fun mean
# python synth.py -c $2 --fun mean

# python join.py -c $1 --suff projection_euclidean projection_dot projection_angle mean
# python join.py -c $2 --suff projection_euclidean projection_dot projection_angle mean

# python process.py -T "data/chunk$1_join.csv" -t "data/chunk$2_join.csv" -u
#
# # python process.py -T data/chunk6_join.csv -t data/chunk7_join.csv -u
#
# python forest.py -T $1 -t $2 --base join_select --stats
#
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun euclidean
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun angle
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun norm_angle
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun vne
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun iqr_euclidean
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun absolute_correlation
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun norm_correlation
# python batch.py -T "data/chunk$1_join_select.csv"  -t "data/chunk$2_join_select.csv" --col allbut --fun cosine
# # python batch.py -T "data/chunk$1_join_select.csv" -t "data/chunk$1_join_select.csv" --
