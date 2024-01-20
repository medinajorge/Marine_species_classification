#! /bin/bash

#for weather in none
#do for v in arch-segment
#do for o in none space
#do for mode in random species stratified
#do    
#    percentages="0 0.05"
#    for decimal in $(seq 10 5 95)
#        do percentages+=" 0.$decimal"
#    done
#    for p in $percentages
#        do for seed in $(seq 0 9)
#            do
#            id=tree_"$v"_"$o"_"$weather"_"$mode"_"$p"_"$seed"
#            run -c 1 -m 12 -t 1:10 -o stdin_files/"$id".out -e stdin_files/"$id".err "python species_classification_trajectory_prunning.py -c tree -v $v -o $o -w $weather -M $mode -P $p -s $seed"
#        done
#    done
#done
#done
#done
#done

for weather in none
do for v in arch-segment
do for o in none space
do for mode in random species
do    
    percentages="0.1"
    for decimal in $(seq 2 9)
        do percentages+=" 0.$decimal"
    done
    percentages+=" 0.95"
    for p in $percentages
        do for seed in $(seq 0 4)
            do
            id=inception_"$v"_"$o"_"$weather"_"$mode"_"$p"_"$seed"
            run -c 1 -m 14 -g 1 -t 4:30 -o stdin_files/"$id".out -e stdin_files/"$id".err "python species_classification_trajectory_prunning.py -c inception -v $v -o $o -w $weather -M $mode -P $p -s $seed"
            sleep 20s
        done
    done
done
done
done
done



