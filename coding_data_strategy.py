from politeness.test_documents import TEST_DOCUMENTS, TEST_TEXTS
from politeness.api_util import get_scores_and_strategies

#for text in TEST_TEXTS:
#  print(get_scores_and_strategies(text))

import pandas as pd
import json

df = pd.read_csv("politeness_coding_data.csv", encoding="utf-8", index_col=[0])

for row in df.iterrows():
  