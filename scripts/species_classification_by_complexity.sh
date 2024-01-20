#! /bin/sh

# 2D image
for r in $(seq 1 5)
do
    id=stdin_files/inception_v-true_i-true_r-"$r"
    run -c 1 -m 24 -g 1 -t 2:05 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c inception -v true -i true -r $r"
done

# 2D occupancy
for r in 3 4 #$(seq 1 5)
do
    id=stdin_files/inception_v-true_i-true_d-true_r-"$r"
    run -c 1 -m 24 -g 1 -t 2:05 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c inception -v true -i true -d true -r $r"
done

for r in $(seq 1 5)
do for model in inception xgb
do for v in true false
do
    id=stdin_files/"$model"_v-"$v"_r-"$r"_common_origin_distance-true
    run -c 1 -m 24 -g 1 -t 2:05 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -v $v -r $r -C true"
done
done
done

# Common origin + weather
for r in $(seq 1 5)
do for model in xgb #inception
do for w in all
do
    id=stdin_files/"$model"_w-"$w"_r-"$r"_common_origin_distance-true
    run -c 1 -m 24 -g 1 -t 2:30 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -w $w -r $r -C true"
done
done
done

# pruned env features. First run
for r in 1
do for model in inception
do for w in mrmr #mrmrloop+vif #vif mrmr+vif # pruned mrmr+collinear #all
do
    id=stdin_files/"$model"_w-"$w"_r-"$r"_common_origin_distance-false
    run -c 1 -m 24 -g 1 -t 2:29 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -w $w -r $r -C false"
done
done
done

# pruned env features
for r in $(seq 2 5)
do for model in inception
do for w in mrmr mrmrloop+vif vif mrmr+vif mrmr+collinear pruned all
do
    id=stdin_files/"$model"_w-"$w"_r-"$r"_common_origin_distance-false
    run -c 1 -m 24 -g 1 -t 2:31 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -w $w -r $r -C false"
done
done
done

for r in 1
do for model in resnet
do for d in true false
do
    id=stdin_files/"$model"_d-"$d"_r-"$r"
    run -c 1 -m 24 -g 1 -t 6:30 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -d $d -i true -r $r"
done
done
done

# Not common origin distance or image (with weather=None).
for r in $(seq 1 5)
do for model in inception
do
    id=stdin_files/"$model"_r-"$r"_common_origin_distance-false
    run -c 1 -m 24 -g 1 -t 2:29 -o "$id".out -e "$id".err "python species_classification_by_sample_size.py -c $model -r $r -C false -O true"
done
done
