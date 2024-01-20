#! /bin/bash

for R in not-in-train # binary
do for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
do
    printf -v s "%q" "$s"
    for i in false
    do for w in none all coast-d+bathymetry
    do for m in inception #tree #inception
    do for p in none 6 #3 6 #12 24
    do for C in true false
    do for T in all 0.1 0.2
    do
        id=stdin_files/s-"$s"-w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"_stage-transfer_T-"$T"_R-"$R"
        run -c 1 -m 3 -t 8:0 -o "$id".out -e "$id".err "python stage_transfer.py -w $w -m $m -p $p -C $C -i $i -S $s -T $T -R $R"
    done
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
#         do for T in all 0.1 0.2
#         do
#             id=stdin_files/s-"$s"-w-"$w"_m-"$m"_p-"$p"_C-"$C"_i-"$i"_stage-transfer_T-"$T"_D-"$D"
#             run -c 1 -m 3 -t 8:0 -o "$id".out -e "$id".err "python stage_transfer.py -w $w -m $m -p $p -C $C -i $i -S $s -T $T -D $D"
#         done
#         done
#         done
#         done
#         done
#         done
#     done
# done
