#! /bin/bash

# step=15
# for year in 1985
#     do year_array=($year)
#     for increment in $(seq 1 $(($step - 1)))
#         do year_array+=","$(($year + $increment))
#     done
#     for variable in potential_temperature rotated_meridional_velocity rotated_zonal_velocity salinity
#     do
#         id=stdin_files/"$variable"_"$year"
#         run -c 1 -m 4 -t 30:00 -o $id.out -e $id.err "python nc_data.py -v $variable -y $year_array"
#     done
# done

# step=5
# for year in 2000 2005 2010 2015
#     do year_array=($year)
#     for increment in $(seq 1 $(($step - 1)))
#         do year_array+=","$(($year + $increment))
#     done
#     for variable in potential_temperature rotated_meridional_velocity rotated_zonal_velocity salinity
#     do
#         id=stdin_files/"$variable"_"$year"
#         run -c 1 -m 12 -t 60:00 -o $id.out -e $id.err "python nc_data.py -v $variable -y $year_array"
#     done
# done

for variable in potential_temperature rotated_meridional_velocity rotated_zonal_velocity salinity
do
    id=stdin_files/"$variable"_2019
    run -c 1 -m 6 -t 3:00 -o $id.out -e $id.err "python nc_data.py -v $variable -y 2019 -O true"
done
