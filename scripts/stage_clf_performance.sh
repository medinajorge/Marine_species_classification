#! /bin/bash

# clf=tree
# for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
# do
#     printf -v s "%q" "$s"
#     id=stdin_files/"$clf"_"$s"
#     id_co="$id"_common_origin
#     run -c 1 -m 2 -t 00:10 -o "$id_co".out -e "$id_co".err "python stage_clf_performance.py -c $clf -s $s -w None -C True"

#     id_normal="$id"_normal
#     run -c 1 -m 2 -t 00:10 -o "$id_normal".out -e "$id_normal".err "python stage_clf_performance.py -c $clf -s $s -w None -C False"

#     id_env="$id"_env
#     run -c 1 -m 2 -t 00:10 -o "$id_env".out -e "$id_env".err "python stage_clf_performance.py -c $clf -s $s -w all -C False"
# done

clf=inception
for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
do
    printf -v s "%q" "$s"
    for f in 16 32 64 128
    do
        id=stdin_files/"$clf"_"$s"_"$f"
        id_co="$id"_common_origin
        run -c 1 -m 4 -t 20:00 -o "$id_co".out -e "$id_co".err "python stage_clf_performance.py -c $clf -s $s -w None -C True -f $f"

        id_normal="$id"_normal
        run -c 1 -m 4 -t 20:10 -o "$id_normal".out -e "$id_normal".err "python stage_clf_performance.py -c $clf -s $s -w None -C False -f $f"

        id_env="$id"_env
        run -c 1 -m 5 -t 22:30 -o "$id_env".out -e "$id_env".err "python stage_clf_performance.py -c $clf -s $s -w all -C False -f $f"
    done
done

# clf=resnet
# for s in "Audouins gull" "Corys shearwater" "Northern gannet" "Black-browed albatross" "Bullers albatross" "Grey-headed albatross" "Hawksbill turtle" "Chinstrap penguin" "Scopolis shearwater" "Macaroni penguin" "Wandering albatross" "Red-tailed tropic bird" "Sooty tern" "Baraus petrel" "Wedge-tailed shearwater" "Black-footed albatross" "White-tailed tropic bird"
# do
#     printf -v s "%q" "$s"
#     for f in 32 64 128
#     do
#         id=stdin_files/"$clf"_"$s"_"$f"
#         id_co="$id"_common_origin
#         run -c 1 -m 7 -t 30:00 -o "$id_co".out -e "$id_co".err "python stage_clf_performance.py -c $clf -s $s -w None -C True -f $f"

#         id_normal="$id"_normal
#         run -c 1 -m 7 -t 44:0 -o "$id_normal".out -e "$id_normal".err "python stage_clf_performance.py -c $clf -s $s -w None -C False -f $f"

#         id_env="$id"_env
#         run -c 1 -m 10 -t 60:00 -o "$id_env".out -e "$id_env".err "python stage_clf_performance.py -c $clf -s $s -w all -C False -f $f"
#     done
# done
