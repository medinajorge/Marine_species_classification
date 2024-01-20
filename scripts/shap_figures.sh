#! /bin/bash

for taxa in Bears Birds Sharks Penguins Seals Turtles Whales Seals-Eared Manta Fish Sirenians
do for v in arch-segment
do for o in none space
do for dt in true
do 
    id=stdin_files/taxa-"$taxa"_shap_"$v"_to-origin-"$o"_dt-"$dt"
    run -c 1 -m 12 -t 0:45 -o "$id".out -e "$id".err "python shap_figures.py -v $v -o $o -t $dt -T $taxa"
done
done
done
done