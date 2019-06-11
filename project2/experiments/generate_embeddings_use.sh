#!/bin/bash

FILE="story_cloze.generate_embeddings"
bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE \
    --file stories.eval.csv \
    --mode eval \
    --encoder_type universal \

bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE \
    --file stories.spring2016.csv \
    --mode test \
    --encoder_type universal \
