
## Project 1 - Natural Language Understanding 2019

### Group 22: Akmaral Yessenalina, Vignesh Ram Somnath, Ritu Sriram, Meet Vora


The project assumes the following structure:
```
data/
    sentences.train
    sentences_test.txt
    sentences.eval
    sentences.continuation
language_model.py
experiments.py
dataset.py
load_embeddings.py
results/
logs/
models/
```
Trained models are saved under `models/`, result files under `result/` and Tensorboard logs under `logs/`

To run any of the experiments A, B or C, run:
```
python3 -m experiments --exp_type X where X in {a, b, c}
```

To save perplexities, run:
```
python3 -m experiments --exp_type X --restore_name EXP_NAME --restore_epoch EPOCH_NUM
```

where `EXP_NAME` is the timestamped name of experiment (directory name under `models/` or `logs/`) and `EPOCH_NUM` is the best perfoming epoch according to eval dataset

For conditional generation, run:
```
python3 -m experiments --exp_type d --restore_name Experiment_C_XX_XX_XX --restore_epoch EPOCH_NUM
```

