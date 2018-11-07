import pandas as pd
import pickle
import random
import numpy as np

def _read(feature_path="train_features.pkl", label_path="train.csv"):
  with open(feature_path, "rb") as fin:
    pkl = pickle.load(fin)
  df = pd.read_csv(label_path, index_col=0)
  X = pkl.toarray()
  Y = df.loc[:, "ch_mean":"en_mean"].values
  return np.concatenate((X, Y), axis=1)

def read(feature_path="train_features.pkl", label_path="train.csv"):
  # arr = _read(feature_path, label_path)
  # return {
  #   "X": arr[:, :-2]
  #   "Y": arr[:, -2:]
  # }
  return _read(feature_path, label_path)