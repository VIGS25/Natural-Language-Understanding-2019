#!/bin/bash

FILE="story_cloze.train.train_ffn"
bsub -n 20 -N -W 20:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE \
    --max_checkpoints_to_keep 10 \
    --batch_size 64 \
    --rnn_type lstm \
    --input_mode full_context \
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
