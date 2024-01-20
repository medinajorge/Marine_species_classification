#! /bin/bash

for r in median #median
do
    # id_large=stdin_files/shap_v2_large_"$r"
    # run -c 1 -m 2 -t 3:00 -o "$id_large".out -e "$id_large".err "python occurrences_count.py -r $r -l 1 -L 1"

    id_short=stdin_files/shap_v2_short_"$r"
    run -c 1 -m 2 -t 8:00 -o "$id_short".out -e "$id_short".err "python occurrences_count.py -r $r -l 0.5 -L 0.5"
done

for r in None
do
    # id_large=stdin_files/shap_v2_large_"$r"
    # run -c 1 -m 2 -t 3:00 -o "$id_large".out -e "$id_large".err "python occurrences_count.py -r $r -l 1 -L 1"

    id_short=stdin_files/shap_v2_short_"$r"
    run -c 1 -m 2 -t 8:00 -o "$id_short".out -e "$id_short".err "python occurrences_count.py -r $r -l 0.5 -L 0.5"
done
