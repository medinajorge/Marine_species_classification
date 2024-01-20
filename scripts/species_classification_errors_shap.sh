#! /bin/bash

for r in 3
do for s in true false
do
    id=stdin_files/errors_shap_weather_r-"$r"_s-"$s"
    run -c 1 -m 4 -t 2:00 -e "$id".err -o "$id".out "python species_classification_errors_shap.py -w all -C false -r $r -s $s -O true"
    id=stdin_files/errors_shap_common_origin_r-"$r"_s-"$s"
    run -c 1 -m 4 -t 2:00 -e "$id".err -o "$id".out "python species_classification_errors_shap.py -w None -C true -r $r -s $s -O true"
done
done
