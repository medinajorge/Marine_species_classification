#! /bin/bash

for R in not-in-train #binary
do for S in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
do
    printf -v S "%q" "$S"
    for i in false
    do for w in none all coast-d+bathymetry
    do for p in none 6 #3 6 #12 24
    do for C in true false
    do
        id=stdin_files/s-"$S"-w-"$w"_p-"$p"_C-"$C"_i-"$i"_stage-transfer-zero-shot_R-"$R"
        run -c 1 -m 2 -t 1:0 -o "$id".out -e "$id".err "python stage_transfer_zero-shot.py -w $w -p $p -C $C -i $i -S $S -R $R"
    done
    done
    done
    done
done
done

# for S in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
# do
#     printf -v S "%q" "$S"
#     for D in "cos t,sin t"
#     do
#         printf -v D "%q" "$D"
#         for i in false
#         do for w in none all coast-d+bathymetry
#         do for p in none 6 #3 6 #12 24
#         do for C in true false
#         do
#             id=stdin_files/s-"$S"-w-"$w"_p-"$p"_C-"$C"_i-"$i"_stage-transfer-zero-shot_D-"$D"
#             run -c 1 -m 2 -t 1:0 -o "$id".out -e "$id".err "python stage_transfer_zero-shot.py -w $w -p $p -C $C -i $i -S $S -D $D"
#         done
#         done
#         done
#         done
#     done
# done
