#! /bin/bash

#for weather in none #all
#do for v in none
#do for o in none space #all
#do run -c 1 -m 12 -t 1:20 -o stdin_files/tree_"$v"_"$o"_"$weather".out -e stdin_files/tree_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c tree -v $v -o $o -w $weather"
#done
#done
#done
#
#
#for weather in none #all
#do for v in none
#do for o in none space #all
#do run -c 12 -m 100 -t 4:0 -o stdin_files/forest_"$v"_"$o"_"$weather".out -e stdin_files/forest_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c forest -v $v -o $o -w $weather"
#done
#done
#done

for weather in none #all
do for v in none
do for o in none  #space
do for model in dense #fcn conv_lstm deep_conv_lstm
do run -c 1 -m 24 -g 1 -t 4:0 -o stdin_files/"$model"_"$v"_"$o"_"$weather".out -e stdin_files/"$model"_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c $model -v $v -o $o -w $weather"
done
done
done
done

#for weather in none #all
#do for v in none
#do for o in none space #all
#do for model in inception lstm
#do run -c 1 -m 24 -g 1 -t 4:30 -o stdin_files/"$model"_"$v"_"$o"_"$weather".out -e stdin_files/"$model"_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c $model -v $v -o $o -w $weather"
#done
#done
#done
#done

#for weather in none #all
#do for v in none
#do for o in none space #all
#do for model in resnet 
#do run -c 1 -m 24 -g 1 -t 24:0 -o stdin_files/"$model"_"$v"_"$o"_"$weather".out -e stdin_files/"$model"_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c $model -v $v -o $o -w $weather"
#done
#done
#done
#done


#for weather in none #all
#do for v in none
#do for o in none space #all
#do for model in xgb 
#do run -c 1 -m 24 -g 1 -t 3:0 -o stdin_files/"$model"_"$v"_"$o"_"$weather".out -e stdin_files/"$model"_"$v"_"$o"_"$weather".err "python density_vs_acc.py -c $model -v $v -o $o -w $weather"
#done
#done
#done
#done

