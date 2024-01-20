#! /bin/bash

for year in 1985 #$(seq 1985 1986)
do run -c 1 -m 30 -o stdin_files/grib_pruner_"$year".out -e stdin_files/grib_pruner_"$year".err "julia Grib_pruner.jl -y $year"
done