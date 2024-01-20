#! /bin/bash

# for i in true false
# do for w in coast-d+bathymetry # none
# do for m in inception # tree #inception
# do for p in 6 none 1 #3 6 #12 24
# do for C in true # false
# do
#     id=stdin_files/w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"
#     run -c 1 -m 5 -t 20:00 -o "$id".out -e "$id".err "python stage_clf_report.py -w $w -m $m -p $p -C $C -i $i"
# done
# done
# done
# done
# done

for i in true false
do for w in none all #coast-d+bathymetry
do for m in inception #tree inception
do for p in none #1 #3 6 #12 24
do for C in false #true
do
    id=stdin_files/w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"
    run -c 1 -m 5 -t 20:00 -o "$id".out -e "$id".err "python stage_clf_report.py -w $w -m $m -p $p -C $C -i $i"
done
done
done
done
done


for i in true #false
do for D in "cos t,sin t"
do
    printf -v D "%q" "$D"
    for w in coast-d+bathymetry #all none
    do for m in inception # tree #inception
    do for p in 1 # 6 #3 12 24 none
    do for C in false #true
    do
        id=stdin_files/w-"$w"_m-"$m"_p-"$p"_C-"$C"_D-"$D"_i-"$i"
        run -c 1 -m 5 -t 20:00 -o "$id".out -e "$id".err "python stage_clf_report.py -w $w -m $m -p $p -C $C -D $D -i $i"
    done
    done
    done
    done
done
done


for i in false # true
do for D in "cos t,sin t"
do
    printf -v D "%q" "$D"
    for w in  coast-d+bathymetry #all none
    do for m in inception # tree #inception
    do for p in none #1 6 #3 12 24
    do for C in true #false
    do
        id=stdin_files/w-"$w"_m-"$m"_p-"$p"_C-"$C"_D-"$D"_i-"$i"
        run -c 1 -m 5 -t 20:00 -o "$id".out -e "$id".err "python stage_clf_report.py -w $w -m $m -p $p -C $C -D $D -i $i"
    done
    done
    done
    done
done
done
