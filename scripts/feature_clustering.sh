#! /bin/bash

# for m in loss #accuracy
# do for N in 50000 #200000
# do
#     id=stdin_files/feature_clustering_m-"$m"_N-"$N"
#     run -c 1 -g 1 -m 24 -t :20 -o "$id".out -e "$id".err "python feature_clustering.py -m $m -N $N"
# done
# done

for m in accuracy loss
do for N in 300000 None
do
    id=stdin_files/feature_clustering_m-"$m"_N-"$N"
    run -c 1 -g 1 -m 24 -t 1:00 -o "$id".out -e "$id".err "python feature_clustering.py -m $m -N $N"
done
done
