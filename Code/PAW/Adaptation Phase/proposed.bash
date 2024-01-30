#!/bin/bash

echo which dataset?
read dataset

echo gpu id?
read gpu_id

for random_seed in {0..4}
do
  #echo ca_2a$random_seed$dataset
  python adaptation.py --name 0727_$dataset$random_seed --seed $random_seed --dataset $dataset --gpu_id $gpu_id --base_model eegtcnet
done