#! /bin/bash

# for N in 0 #5 10 25 50 100 200
# do for r in $(seq 1 5)
# do for S in "Loggerhead turtle" "Leatherback turtle" "Hawksbill turtle" "Green turtle"
# do
#     printf -v S "%q" "$S"
#     id=stdin_files/"${S}"_N-"${N}"_r-"${r}"
#     run -c 1 -m 24 -g 1 -t 1:30 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -w all -C false -c inception"
# done
# done
# done

# # ADDED WEATHER, common_origin_distance False
# for S in "Audouins gull" "Corys shearwater" "Northern gannet" "Little penguin" "Black-browed albatross" "Southern elephant seal" "Bullers albatross" "Grey-headed albatross" "Australian sea lion" "Macaroni penguin" "Loggerhead turtle" "Chinstrap penguin" "Scopolis shearwater" "Blue shark" "Shortfin mako shark" "Whale shark" "Tiger shark" "Leatherback turtle" "Wandering albatross" "Northern elephant seal" "Humpback whale" "Salmon shark" "Short-finned pilot whale" "Adelie penguin" "King eider" "Hawksbill turtle" "White shark" "California sea lion" "Masked booby" "Green turtle" "Long-nosed fur seal" "Blue whale" "Black-footed albatross" "Trindade petrel" "Laysan albatross" "Ringed seal"
# do
#     printf -v S "%q" "$S"
#     for N in 5 10 22 46 96 200
#     do for r in 4 5 # $(seq 1 3)
#     do
#         id=stdin_files/"${S}"_N-"${N}"_r-"${r}"_weather
#         run -c 1 -m 24 -g 1 -t 2:11 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -C false -c inception -V true -w all"
#     done
#     done
# done

# # ADDED WEATHER, common_origin_distance False
# for S in "Trindade petrel" "Laysan albatross" "Ringed seal"
# do
#     printf -v S "%q" "$S"
#     for N in 5 10 22 46 96 200
#     do for r in 4 5 # $(seq 1 3)
#     do
#         id=stdin_files/"${S}"_N-"${N}"_r-"${r}"_weather
#         run -c 1 -m 24 -g 1 -t 2:11 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -C false -c inception -V true -w all"
#     done
#     done
# done

# # ADDED WEATHER, common_origin_distance False
# for S in "Black-browed albatross" "Southern elephant seal" "Bullers albatross" "Grey-headed albatross" "Australian sea lion" "Macaroni penguin" "Loggerhead turtle" "Chinstrap penguin" "Scopolis shearwater" "Blue shark" "Shortfin mako shark" "Whale shark" "Tiger shark" "Leatherback turtle" "Wandering albatross" "Northern elephant seal" "Humpback whale" "Salmon shark" "Short-finned pilot whale" "Adelie penguin" "King eider" "Hawksbill turtle" "White shark" "California sea lion" "Masked booby" "Green turtle" "Long-nosed fur seal" "Blue whale" "Black-footed albatross" "Trindade petrel" "Laysan albatross" "Ringed seal"
# do
#     printf -v S "%q" "$S"
#     for N in 5 10 22 46 96 200
#     do for r in 4 5 # $(seq 1 3)
#     do
#         id=stdin_files/"${S}"_N-"${N}"_r-"${r}"_weather
#         run -c 1 -m 24 -g 1 -t 2:11 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -C false -c inception -V true -w all"
#     done
#     done
# done

for S in "Audouins gull" "Corys shearwater" "Northern gannet" "Little penguin" "Black-browed albatross" "Southern elephant seal" "Bullers albatross" "Grey-headed albatross" "Australian sea lion" "Macaroni penguin" "Loggerhead turtle" "Chinstrap penguin" "Scopolis shearwater" "Blue shark" "Shortfin mako shark" "Whale shark" "Tiger shark" "Leatherback turtle" "Wandering albatross" "Northern elephant seal" "Humpback whale" "Salmon shark" "Short-finned pilot whale" "Adelie penguin" "King eider" "Hawksbill turtle" "White shark" "California sea lion" "Masked booby" "Green turtle" "Long-nosed fur seal" "Blue whale" "Black-footed albatross" "Trindade petrel" "Laysan albatross" "Ringed seal"
do
    printf -v S "%q" "$S"
    for N in 5 10 22 46 96 200
    do for r in 4 5 #$(seq 1 3)
    do
        id=stdin_files/"${S}"_N-"${N}"_r-"${r}"_common_orig
        run -c 1 -m 24 -g 1 -t 2:15 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -C true -c inception -V true"
    done
    done
done


# # ADDED WEATHER, common_origin_distance False
# for S in "Audouins gull"
# do
#     printf -v S "%q" "$S"
#     for N in 5 10 22
#     do for r in $(seq 1 3)
#     do
#         id=stdin_files/"${S}"_N-"${N}"_r-"${r}"
#         run -c 1 -m 24 -g 1 -t 2:10 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -S "$S" -N $N -r $r -C false -c inception -V true -w all"
#     done
#     done
# done
