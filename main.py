from politeness.test_documents import TEST_DOCUMENTS, TEST_TEXTS
from politeness.api_util import get_scores_and_strategies

for text in TEST_TEXTS:
  print(get_scores_and_strategies(text))