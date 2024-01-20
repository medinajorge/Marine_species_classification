#! /bin/bash

# day and len
#for m in 1
#do for l in 60 100 200
#do for W in none linear
#do 
#    id=stdin_files/days-"$m"_len-"$l"_W-"$W"_scale
#    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -m $m -l $l -W $W -s true"
#done
#done
#done

#for m in 1 3 10
#do for l in 30 60 100 200
#do 
#    id=stdin_files/days-"$m"_len-"$l"
#    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -m $m -l $l"
#done
#done
#
#for m in 10
#do for l in 200
#do 
#    id=stdin_files/days-"$m"_len-"$l"
#    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -m $m -l $l"
#done
#done
#
#
## diff and scaling 
#for D in true false
#do for s in true none
#do
#    id=stdin_files/scale-"$s"_diff-"$D"_bat
#    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -s $s -D $D -b true"
#done
#done

#features
#for w in none temperature wind waves all
#do
#    id=stdin_files/weather-"$w"_bat
#    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -b true -w $w"
#done

# class weights
#id=stdin_files/weights
#run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python stage_classification.py -W none"

#id=stdin_files/v_and_t
#run -c 1 -m 24 -g 1 -t 0:15 -o "$id".out -e "$id".err "python stage_classification.py -d x,y,z -w none"

# only movement
#id=stdin_files/v_only
#run -c 1 -m 24 -g 1 -t 0:15 -o "$id".out -e "$id".err "python stage_classification.py -d x,y,z,cos-t,sin-t -w none"
#run -c 1 -m 24 -g 1 -t 0:20 -o "$id"_bat.out -e "$id"_bat.err "python stage_classification.py -d x,y,z,cos-t,sin-t -w none -b true"

# only movement
#id=stdin_files/v_only_scaled
#run -c 1 -m 24 -g 1 -t 0:15 -o "$id".out -e "$id".err "python stage_classification.py -d x,y,z,cos-t,sin-t -w none -s true"
#run -c 1 -m 24 -g 1 -t 0:20 -o "$id"_weather.out -e "$id"_weather.err "python stage_classification.py -d x,y,z,cos-t,sin-t -s true"

#id=stdin_files/only_time
#run -c 1 -m 24 -g 1 -t 0:15 -o "$id".out -e "$id".err "python stage_classification.py -d x,y,z -w none -s true -v none"

for attr in stage
do for weather in none
do for del in none sin-t,cos-t
do for v in none x-y-z
do 
    id=stdin_files/"$attr"_weather-"$weather"_del-"$del"-v_"$v"
    run -c 1 -m 24 -g 1 -t 0:20 -o "$id".out -e "$id".err "python "$attr"_classification.py -w $weather -d $del -v $v -s true"
done
done
done
done
