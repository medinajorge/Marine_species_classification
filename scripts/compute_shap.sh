#! /bin/bash

#for v in arch-segment #x-y-z
#do for o in space #none
#do run -c 1 -m 24 -g 1 -t 15:30 -o stdin_files/shap_"$v"_to-origin-"$o".out -e stdin_files/shap_"$v"_to-origin-"$o".err "python compute_shap.py -v $v -o $o -t true"
#done
#done

for taxa in Dolphins #Bears Birds Sharks Penguins Seals Turtles Whales Seals-Eared Manta Fish Sirenians 
do for v in arch-segment #x-y-z
do for o in none space
do 
    id=shap_"$v"_to-origin-"$o"_taxa-"$taxa"
    run -c 1 -m 10 -g 1 -t 1:0 -o stdin_files/"$id".out -e stdin_files/"$id".err "python compute_shap.py -v $v -o $o -t true -T $taxa"
done
done
done