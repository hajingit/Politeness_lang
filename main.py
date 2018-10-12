from politeness.test_documents import TEST_DOCUMENTS, TEST_TEXTS
from politeness.api_util import get_scores_strategies_token_indices

import pprint

# for text in TEST_TEXTS:
#   ret = get_scores_strategies_token_indices("")
#   pprint.pprint(ret)
ret = get_scores_strategies_token_indices("What are you? I am fine.")
pprint.pprint(ret)