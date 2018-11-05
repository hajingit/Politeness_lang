import pandas as pd

def read(path="train_strategy.csv"):
  df = pd.read_csv(path, index_col=0)
  X = df.loc[:, "2nd person":].values
  Y = df.loc[:, "ch_mean":"en_mean"].values
  return {
    "X": X,
    "Y": Y,
  }