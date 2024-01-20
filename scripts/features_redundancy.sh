#! /bin/bash

for f in "x" "y" "z" "sin t" "cos t" "Marine current (meridional)" "Marine current (zonal)" "Salinity" "Wind (U)" "Wind (V)" "Pressure" "Temperature" "Wave period" "Wave height" "Precipitation" "K index" "Sea ice fraction" "Solar radiation" "Thermal radiation" "IR albedo (diffuse)" "IR albedo (direct)" "UV radiation" "Evaporation" "Geopotential" "Wave direction (x)" "Wave direction (y)" "Distance to coast" "Bathymetry" "Marine current (meridional) 97m" "Marine current (zonal) 97m" "Temperature 97m" "Salinity 97m" "Marine current (meridional) 1516m" "Marine current (zonal) 1516m" "Temperature 1046m" "Temperature 10m" "Salinity 10m"
do
    printf -v f "%q" "$f"
    id=stdin_files/features_redundancy_"$f"
    run -c 1 -m 4 -t 8:00 -o "$id".out -e "$id".err "python features_redundancy.py -f $f"
done
