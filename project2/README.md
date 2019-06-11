# Project 2

## Project structure

* `data/` - Data folder. Contains scripts to download data.
* `report/` - The LaTeX report and the generated PDF.
* `story_cloze/` - Source Folder
  * `embeddings/` -
    * `skip_thoughts/` - External module from [TensorFlow-Models](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
    * `sentence_encoders.py` - Implementations of different sentence encoders used in the experiments.
  * `models/` - Contains implementations of different models used in the experiments
    * `base_model.py` - BaseModel class, which other models inherit from. Contains functionality for initial setup, fit and evaluate
    * `ffn.py` - FeedForward Network with different contextual encodings, as implemented in [Srinivasan et al.](https://arxiv.org/abs/1803.05547)
    * `roemmele_rnn.py` - RNN model based on [Roemmele et al.](https://www.aclweb.org/anthology/W17-0911)
    * `roemmele_birnn.py` - BiRNN model extending the original implementation in [Roemmele et al.](https://www.aclweb.org/anthology/W17-0911)
  * `train/` - Contains the train scripts for each of the models
  * `datasets.py` - Functionality for loading and encoding and batching datasets
  * `attention.py` - Implementation of the attention modules
  * `generate_SkipThoughts.py` - Generate and save SkipThoughts embeddings for any dataset (preferably the validation or any test set). The SkipThoughts embeddings for the original training set are generated
      the first time when running any of the train scripts. For the other times, it is loaded. (This takes ~ 7hrs though)
  * `generate_USE.py` - Generate and save Universal Sentence Encoder (USE) embeddings for train, validation and original Story Cloze test dataset.
     Since we don't use USE for predictions, there is no script to generate USE embeddings for a custom file.
* `experiments/` - Contains bash scripts for each of the experiments in the report.
* `Makefile` - Contains definitions for various tasks.
* `README.md` - This manual
* `requirements.txt` - Required Python packages.
* `final_predictions.csv` - Final Predictions
* `generate_final_predictions.csv` - Generate the Final Predictions

## Set up

* Install Python 3.6+.
* Ensure `SCRATCH` variable is set to a folder with sufficient memory for pre-trained models, train, validation and test embeddings (~100GB), model checkpoints and logs (~15GB).
* Run the setup:
    ```
    make requirements
    ```
* model checkpoints and logs are saved under `$SCRATCH/outputs/EXP_NAME`

## SkipThoughts Based Experiments

### [Roemmele et al.](https://www.aclweb.org/anthology/W17-091)

All experiments based on Roemmele et al are prefixed with `roemmele_`.
* To run experiments without attention use the command:

```
make roemmele_{RNN_CELL}
```

where `RNN_CELL` is one of `{gru, lstm, vanilla, bigru, bilstm}`

e.g. To run the standard Roemmele GRU experiment, the command would be:

```
make roemmele_gru
```

* To run experiments with attention, use the command:

```
make roemmele_{RNN_CELL}_{ATTN_TYPE}_attn
```

where `RNN_CELL` is as described above, and `ATTN_TYPE` is one of `{mult, add}`

e.g. To run the Roemmele GRU with multiplicative attention, the command would be:

```
make roemmele_gru_mult_attn
```
### [Srinivasan et al.](https://arxiv.org/abs/1803.05547)

All experiments based on Srinivasan et al. are prefixed with `srini_ffn_`

* To run experiments using the full context mode, use the command:

```
make srini_ffn_fc_{RNN_CELL}
```

where `RNN_CELL` is one of `{gru, lstm}`


* To run experiments using the last sentence based context, use the command:

```
make srini_ffn_ls
```

The above instructions are for running experiments on the original training set.
Since we use validation data based experiments only for comparison, all these experiments are pooled into a single script.
To run experiments using validation set, first use the following command if the embeddings are not saved:

```
make generate_embeddings_st
```
Once the job above is finished (about 5 mins), run:

```
make skipthoughts_all_val
```

## Universal Sentence Encoder (USE) Based Experiments

Since the focus of our work is SkipThoughts, and the results they achieve are significantly better than USE,
we pool all experiments using USE into a single bash script.

To generate all embeddings, first run the command:

```
make generate_embeddings_use
```

and wait for the job to finish.

Then, to run all the experiments on the training set using USE, use the command:

```
make universal_all_train
```

To run all the experiments on the validation set using USE, use the command:

```
make universal_all_val
```

## Saving and Restoring models

Models are currently saved whenever a new best validation score is obtained.
How often validation is performed is  controlled by the `eval_every` argument,
which can be changed in the bash script corresponding to the `make` command.

NOTE: Predictions and restore are currently applicable only to SkipThoughts based models.

The restore script assumes two things:

1) An lsf file is present in the directory you restore from. This is used for
getting the arguments to load the model.

2) The embeddings of the file to evaluate or predict are saved in `$SCRATCH/data`.
For evaluation, this would take the form of `eval_embeddings_SkipThoughts_<ENCODER_MODE>.npy`,
while for prediction, this would take the form of `test_embeddings_SkipThoughts_<ENCODER_MODE>.npy`

where `<ENCODER_MODE>` is in `{uni, bi, both}`, `both` used by default.

To save embeddings, use the command:

```
python -m story_cloze.generate_SkipThoughts --mode {eval/test} --file <FILENAME>
```

To restore a model for evaluation on a different dataset, please use:

```
python -m restore --mode evaluate --restore_from <PATH_TO_CHECKPOINT> --model_name <MODEL_NAME> --test_file <FILENAME>
```

To generate predictions, please use:

```
python -m restore --mode predict --restore_from <PATH_TO_CHECKPOINT> --model_name <MODEL_NAME> --test_file <FILENAME>
```

The predictions are saved to the same location the model was restored from.

## Generating Final Predictions

To generate our final predictions, first run:

```
python -m story_cloze.generate_SkipThoughts --mode test --file stories.test.csv
```

Then, download the Best-Models file from [here](https://polybox.ethz.ch/index.php/s/3h8DX79ZUyCO9gn) and extract it to your root folder (where `generate_final_predictions.py`) is present.

After extraction the structure should look something like:
* Best-Models\
    * FFN-Last-Sentence\
    * Roemmele-BiGRU\
    * Roemmele-BiGRU-Add-Attn\
*generate_final_predictions.py`

Now run,

```
python -m generate_final_predictions
```
