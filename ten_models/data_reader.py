import pandas as pd
import pickle

def read(feature_path="train_features.pkl", label_path="train.csv"):
  with open(feature_path, "rb") as fin:
    pkl = pickle.load(fin)
  df = pd.read_csv(label_path, index_col=0)

  return {
    "X": pkl,
    "Y": df.loc[:, "ch_mean":"en_mean"].values
  }