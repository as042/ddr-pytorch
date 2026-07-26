# ddr-pytorch

A small PyTorch project that predicts the official difficulty rating of a
Dance Dance Revolution chart from its step-pattern statistics. The raw chart spreadsheet contains a
few hundred columns, so `clean_dataframe` first strips out the ones that would leak the answer
(existing difficulty estimates, groove radar values) along with scoring junk and non-numeric fields,
leaving numeric features such as note counts, BPM range, crossovers, footswitches, jacks, brackets,
and notes-per-second statistics. Those features are standardized with a scikit-learn
`StandardScaler` and fed to a small fully connected regression network (`DDRModel`), trained with
Adam and a smooth L1 loss. Everything is driven through one argparse CLI with three subcommands:
`clean_data`, `train`, and `run`. The scaler is saved alongside the model weights so that inference
applies the exact same feature scaling used during training.

## Example

```bash
# 1. Clean a raw chart spreadsheet into a numeric training set.
python source/cli.py clean_data "data/DanceDanceRevolution Chart Data Spreadsheet - Single.csv" data/ddr_clean.csv

# 2. Train a model. Weights and the fitted scaler are written to output.pt.
python source/cli.py train data/ddr_clean.csv --epochs 80 --batch-size 32 --lr 0.001

# 3. Predict the rating of a single chart, by row index in the dataset.
python source/cli.py run models/ddr_model_v2_e160.pt data/ddr_clean.csv 42
```

The `run` command prints the true rating followed by the model's guess, so output looks roughly
like this (120 features remain after the identifying columns and the target are dropped):

```
Loaded model from models/ddr_model_v2_e160.pt
Final feature count: 120
y: 1.0
Guess: 1.2643...
```
