#!/bin/bash -x

seed=$1
selector=$2
lambdc=$3
lambds=$4
dataset=$5
epochs=$6
batch=$7
clients=$8
red=$9

python -um mimic3models.in_hospital_mortality.feature_select_l21 --num_clients $clients --network mimic3models/keras_models/dense_bottom.py --dim 64 --timestep 1.0 --depth 3 --mode train --seed $seed --lr 0.01 --batch_size $batch --output_dir mimic3models/in_hospital_mortality --epochs $epochs --local_epochs 1 --selector $selector --dataset $dataset --lambda_client $lambdc --lambda_server $lambds --optimizer Adam --redundancy $red 
