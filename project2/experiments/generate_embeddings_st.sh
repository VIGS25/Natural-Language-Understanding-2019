#!/bin/bash

FILE="story_cloze.generate_SkipThoughts"
bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE \
    --file stories.eval.csv \
    --mode eval \
    --embed_mode both \

bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE \
    --file stories.spring2016.csv \
    --mode test \
    --embed_mode both \
