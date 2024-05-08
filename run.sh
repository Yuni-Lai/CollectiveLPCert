#!/usr/bin/env bash

#nohup bash run.sh > ./run.log 2>&1 &


#datset=['cora', 'citeseer', 'pubmed']

model='GCN'
### grid search###
for seed in 2021 2022 2023 2024 2025
do
  for optimization in 'LP1' 'LP2'
  do
    for dataset in 'citeseer' 'cora'
    do
      for p_e in 0.7 0.8 0.9 #0.5 0.6 0.7 0.8 0.9
      do
        for p_n in 0.7 0.8 0.9
        do
          echo "The current program (dataset,seed,optimization,p_e,p_n) is: ${dataset},${seed},${optimization},${p_e},${p_n}"
          nohup python -u main.py -dataset $dataset -p_e $p_e -p_n $p_n -optimization $optimization -seed $seed > ./results_${dataset}_${model}/evasion_mode_include/run_${p_e}_${p_n}_${optimization}.log 2>&1 &
        done
        wait
      done
    done
  done
done

echo "Proccess Finished!"
