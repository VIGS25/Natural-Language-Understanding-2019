#!/bin/bash

bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m train_rnn \
    --batch_size 100 \
    --rnn_type gru \
    --num_hidden_units 1000 \
    --encoder_type skipthoughts \
    --embed_mode both \
    --clip_norm 10 \
    --story_length 4 \
    --n_random 6 \
    --learning_rate 1e-3 \
    --num_epochs 20 \
    --log_every 1000 \
    --print_every 1000 \
    --eval_every 1000
