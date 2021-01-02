#!/bin/bash
set -e
set -x

python train_tabnet.py \
       --csv-path data/odds.csv \
       --target-name "won" \
       --categorical-features sport,championat,league,E,status \
       --task classification \
       --val-frac 0.2 \
       --test-frac 0.1 \
       --model-name tabnet_adult_census \
       --tb-log-location adult_census_logs \
       --emb-size 1 \ 
       --feature_dim 16 \
       --output_dim 16 \
       --n_steps 5 \
       --lambda-sparsity 0.0001 \
       --gamma 1.5 \
       --lr 0.02 \
       --batch-momentum 0.98 \
       --batch-size 4096 \
       --virtual-batch-size 128 \
       --decay-every 500 \
       --max-steps 3000
