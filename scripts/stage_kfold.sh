#! /bin/bash

# for i in false
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


# STAGE REMAP
for R in not-in-train # binary
do for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
do
    printf -v s "%q" "$s"
    for i in false
    do for w in none all coast-d+bathymetry
    do for m in inception #tree #inception
    do for p in none 6 #3 6 #12 24
    do for C in false true
    do
        id=stdin_files/s-"$s"-w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"_stage-kfold_R-"$R"
        run -c 1 -m 4 -t 3:0 -o "$id".out -e "$id".err "python stage_kfold.py -w $w -m $m -p $p -C $C -i $i -S $s -R $R"
    done
    done
    done
    done
    done
done
done

# # Delete time vars
# for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
# do
#     printf -v s "%q" "$s"
#     for D in "cos t,sin t"
#     do
#         printf -v D "%q" "$D"
#         for i in false
#         do for w in none all coast-d+bathymetry
#         do for m in inception #tree #inception
#         do for p in none 6 #3 6 #12 24
#         do for C in true # false
#         do
#             id=stdin_files/s-"$s"-w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"_stage-kfold_T_D-"$D"
#             run -c 1 -m 4 -t 3:0 -o "$id".out -e "$id".err "python stage_kfold.py -w $w -m $m -p $p -C $C -i $i -S $s -D $D"
#         done
#         done
#         done
#         done
#         done
#     done
# done
