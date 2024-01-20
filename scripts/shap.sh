#! /bin/bash

for i in $(seq 1 5)
do for w in all # none
do
    id=stdin_files/shap_v2_"$i"_"$w"
    run -g 1 -c 1 -m 24 -t 3:00 -o "$id".out -e "$id".err "python shap.py -i $i -w $w -C true"
done
done
