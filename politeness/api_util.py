from .model import score_and_strategies
from .features.vectorizer import PolitenessFeatureVectorizer

def get_scores_and_strategies(msg):
  for doc in PolitenessFeatureVectorizer.preprocess([msg]):
    probs, strategies = score_and_strategies(doc)
    return {
      "score_polite"    : probs['polite'],
      "score_impolite"  : probs["impolite"],
      "strategies"      : strategies
    }



if __name__ == "__main__":
  from test_documents import TEST_DOCUMENTS, TEST_TEXTS

  for text in TEST_TEXTS:
    print(get_scores_and_strategies(text))