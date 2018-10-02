from politeness.test_documents import TEST_DOCUMENTS, TEST_TEXTS
from politeness.api_util import get_scores_strategies_token_indices

#for text in TEST_TEXTS:
#  print(get_scores_and_strategies(text))

import numpy as np
import pandas as pd
import json

df = pd.read_csv("PolCodings600.csv", encoding="utf-8")

result = {}

for i, row in df.iterrows():
  Num = row["Num"]
  Request = row["Request"]
  ret = get_scores_strategies_token_indices(Request)
  en_mean = np.asscalar(np.nanmean(np.array((row["EN1":"EN31"]).values, dtype=np.float32)))
  ch_mean = np.asscalar(np.nanmean(np.array((row["CH1":"CH21"]).values, dtype=np.float32)))
  for s in ret["strategies"]:
    if s not in result:
      result[s] = []
    result[s].append(
      {
        "Num": Num,
        "Request": Request,
        "en_mean": en_mean,
        "ch_mean": ch_mean,
      }
    )
  if i % 20 == 0:
    print(i)

with open("sort_in_strategies.json", "w", encoding="utf-8") as fout:
  json.dump(result, fout)

import pprint

pprint.pprint(result)