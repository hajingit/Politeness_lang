from politeness.features.vectorizer import PolitenessFeatureVectorizer
ret = PolitenessFeatureVectorizer.preprocess(["what do you think was the purpose of this Fesselballon? if it wasn't military, what was the point in shooting it down?"])
print(ret)