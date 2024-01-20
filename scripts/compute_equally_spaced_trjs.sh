#! /bin/bash

# for w in none
# do for p in 6 12 24 0.5 1 2 3 4
# do
#     id=stdin_files/w-"$w"_p-"$p"
#     run -c 1 -m 5 -t 4:00 -o "$id".out -e "$id".err "python compute_equally_spaced_trjs.py -w $w -p $p"
# done
# done

# for w in coast-d+bathymetry # all
# do for p in 6 12 24 0.5 1 2 3 4
# do
#     id=stdin_files/w-"$w"_p-"$p"
#     run -c 1 -m 16 -t 5:00 -o "$id".out -e "$id".err "python compute_equally_spaced_trjs.py -w $w -p $p"
# done
# done

for w in all
do for p in 24 # 4 6 12 #  24
do
    id=stdin_files/w-"$w"_p-"$p"
    run -c 1 -m 50 -t 8:00 -o "$id".out -e "$id".err "python compute_equally_spaced_trjs.py -w $w -p $p"
done
done
