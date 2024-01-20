#! /bin/bash

# for e in none
# do for n in 3 5 10 20 50
# do
#     id=stdin_files/feature_mi_"$e"_"$n"
#     run -c 1 -m 16 -t 24:00 -o "$id".out -e "$id".err "python features_mi.py -e $e -n $n"
# done
# done

for e in none
do for n in 10
do for S in 6000
do for T in 60000 none
do
    id=stdin_files/feature_mi_"$e"_"$n"_"$S"_"$T"
    run -c 1 -m 8 -t 1:00 -o "$id".out -e "$id".err "python features_mi.py -e $e -n $n -S $S -T $T"
done
done
done
done
