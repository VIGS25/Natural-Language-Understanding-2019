# Project 2

## Project structure

* `data/` - Data folder. Contains scripts to download data.
* `report/` - The LaTeX report and the generated PDF.
* `story_cloze/` - Source Folder
  * `embeddings/` - 
    * `skip_thoughts/` - External module from [TensorFlow-Models](https://github.com/tensorflow/models/tree/master/research/skip_thoughts
    * `sentence_encoders.py` - Implementations of different sentence encoders used in the experiments.
  * `models/` - Contains implementations of different models used in the experiments
    * `base_model.py` - BaseModel class, which other models inherit from. Contains functionality for initial setup, fit and evaluate
    * `ffn.py` - FeedForward Network with different contextual encodings, as implemented in XXX
    * `roemmele_rnn.py` - RNN 
    * `roemmele_birnn.py` - 
  * `train/` - Contains the train scripts for each of the models
  * `datasets.py` - Functionality for loading and encoding and batching datasets
  * `attention.py` - Implementation of the attention modules
* `experiments/` - Contains bash scripts for each of the experiments in the report. Since SkipThoughts were the main focus of our experiments,  we pooled all experiments using 
Universal Sentence Encoder into a single bash script. 
* `Makefile` - Contains definitions for various tasks.
* `README.md` - This manual
* `requirements.txt` - Required Python packages.

## Set up

* Install Python 3.6+.
* Ensure `SCRATCH` environment variable is set to a folder with sufficient memory for pre-trained embeddings (~15GB), storing training/evaluation/testing data, 
model checkpoints and logs (~15GB).
* Run the setup:
    ```
    make requirements
    ```

## Experiments

### Roemmele et al.

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
### Srinivasan et al.

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
