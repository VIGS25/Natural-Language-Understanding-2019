#!/bin/bash

FILE="story_cloze.generate_USE"
bsub -n 20 -N -W 10:00 -R "rusage[mem=10240, ngpus_excl_p=1]" python -m $FILE
