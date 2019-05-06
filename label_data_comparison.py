import sys
import math
import pandas as pd
import vaderSentiment
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
#from vaderSentiment import SentimentIntensityAnalyzer
import spacy
import requests
import json
import sys
import os
from itertools import zip_longest
from sklearn import linear_model
from scipy import sparse
import pickle
import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import politeness.api_util
from politeness.api_util import get_scores_strategies_token_indices

#global variables:
#used in logistic regression
L2_REGULARIZATION_STRENGTH = 0.9

#headers and parameters for perspective api call
headers = {
    'Content-Type': 'application/json',
}

params = (
    ('key', 'AIzaSyBaMPpybrBfyWF54hvkFK1QuEBPPKmQh8M'),
)

def readData(file_name):
    """
    Reads from the data file and returns data frame
    :param file_name: reads the file name
    :return: return a data frame read from file
    """
    data = pd.read_csv(file_name)
    return data

def feature_encoder(dataobjects):
    """
    Features included in the code are:
    1. sentiment scores: pos, neg and neu
    easiness to read scales:
        2. flesch reading,
        3. dale_chall reading,
        4. gunning_foc score,
        5. smog_index and
        6. text standard scores.
        all these scores are included in the entire feature set
    7. perspective api scores (toxicity scores for the entire text)
    8. politeness score
    9. impolite-ness score
    10. politeness strategies
    11. POS tags

    :param dataobjects: reads the data objects (data frame) which incorporate the text
    :return: a feature encoded matrix of numeric entities for the entire data set
    """

    nlp = spacy.load('en_core_web_sm')
    feature_dict = {}
    feature_set = {}

    cnt=0
    for line in dataobjects:
        if cnt == 0:
            cnt=1
            continue
        feature_dict[cnt]={}
        text = line[2]
        #sentiment scores: scores with pos, neg and neutral scores:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        feature_dict[cnt]['pos']=vs['pos']
        feature_dict[cnt]['neg']=vs['neg']
        feature_dict[cnt]['neu']=vs['neu']
        feature_set['pos']=1
        feature_set['neg']=1
        feature_set['neu']=1

        #easiness to read scores: flesch reading:
        sc = textstat.flesch_reading_ease(text)
        feature_dict[cnt]['easiness']=sc
        feature_set['easiness']=1

        #easiness to read scores: dale chall reading:
        sc = textstat.dale_chall_readability_score(text)
        feature_dict[cnt]['easiness_dale']=sc
        feature_set['easines_dale']=1

        #easiness to read scores: gunning fog reading:
        sc = textstat.gunning_fog(text)
        feature_dict[cnt]['easiness_fog']=sc
        feature_set['easines_fog']=1

        #easiness to read scores: smog index reading:
        sc = textstat.smog_index(text)
        feature_dict[cnt]['easiness_smog']=sc
        feature_set['easines_smog']=1

        #easiness to read scores: text standard reading:
        sc = textstat.text_standard(text, float_output=False)
        feature_dict[cnt]['easiness_standard']=sc
        feature_set['easines_standard']=1

        #preprocessing text to make readable for perspective api scores:
        stry = str(text)
        sent = ''
        for a in stry:
            if a==' ' or (a<='Z' and a>='A') or (a<='z' and a>='a') or (a<='9' and a>='0') or a=='?' or a=='.':
                sent +=a

        #perspective api scores call:
        #comment_string ="""
        data = '{comment: {text:"'+sent+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
        response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=params, data=data)
        j = json.loads(response.text)
        feature_dict[cnt]['toxicity'] =0.0
        try:
            feature_dict[cnt]['toxicity'] = j['attributeScores']['TOXICITY']['summaryScore']['value']
        except:
            try:
                feature_dict[cnt]['toxicity'] = j['attributeScores']['TOXICITY']['summaryScore']['value']
            except:
                try:
                    feature_dict[cnt]['toxicity'] = j['attributeScores']['TOXICITY']['summaryScore']['value']
                except:
                    try:
                        feature_dict[cnt]['toxicity'] = j['attributeScores']['TOXICITY']['summaryScore']['value']
                    except:
                        feature_dict[cnt]['toxicity'] =0.0
        feature_dict[cnt]['toxicity'] =0.0
        feature_set['toxicity']=1

        #politeness strategies and politeness scores features:
        sc = get_scores_strategies_token_indices(text)
        feature_dict[cnt]['score_polite']=sc['score_polite']
        feature_dict[cnt]['score_impolite'] = sc['score_impolite']
        feature_set['score_polite']=1
        feature_set['score_impolite']=1
        #print(feature_dict[cnt]['score_polite'])
        for a in sc['strategies']:
            feature_dict[cnt][a]=1
            feature_set[a]=1

        #POS tags in the text:
        doc = nlp(text)
        for token in doc:
            if (str(token.pos_) not in feature_set):
                feature_set[str(token.pos_)]=1

            if not (str(token.pos_) in feature_dict[cnt]):
                feature_dict[cnt][str(token.pos_)]=1
            else:
                feature_dict[cnt][str(token.pos_)]+=1
        cnt+=1

    #creating a systematic feature matrix from feature set
    feature_matrix = []
    for i in range(1, cnt):
        feature_list = []
        for key in feature_set.keys():
            if key in feature_dict[i]:
                feature_list.append(feature_dict[i][key])
            else:
                feature_list.append(0.0)
        feature_matrix.append(feature_list)

    return feature_matrix

#calling read data
df = readData('Project_Politeness/Batch_Binary_Scores.csv')
df_labels = readData('three_labels_data.csv')

#list of lists of data frame objects, dataobjects: list of list of training modules, labelobjects: list of list of test modules
dataobjects = df.values.tolist()
labelobjects = df_labels.values.tolist()

#getting training feature matrix (english)
feature_train_matrix = feature_encoder(dataobjects)
#getting class labels and appending to feature matrix (english)
Y = []
cnt = 0
for line in dataobjects:
    if cnt==0:
        cnt=1
        continue
    feature_train_matrix[cnt-1].append(line[-2])
    cnt+=1

X = np.array(feature_train_matrix)
Xtrain = np.array(X[:,:-1])
cnt = 0
for line in dataobjects:
    if cnt==0:
        cnt=1
        continue
    Y.append(X[cnt-1][-1])
    cnt+=1

#Fitting logistic regression model over the training data set (english)
log_reg = linear_model.LogisticRegression(C=L2_REGULARIZATION_STRENGTH, penalty='l2', n_jobs=4)
log_reg.fit(Xtrain[:],Y[:])

#Creating feature matrix for test set
feature_label_matrix = feature_encoder(labelobjects)
Xtester = np.array(feature_label_matrix)

#Predicting class labels for dataset: english
YtestEn = log_reg.predict(Xtester[:])

#getting class labels and appending to feature matrix (Chinese)
Y = []
cnt = 0
for line in dataobjects:
    if cnt==0:
        cnt=1
        continue
    feature_train_matrix[cnt-1][-1] = line[-1]
    cnt+=1

X = np.array(feature_train_matrix)
Xtrain = np.array(X[:,:-1])
cnt = 0
for line in dataobjects:
    if cnt==0:
        cnt=1
        continue
    Y.append(X[cnt-1][-1])
    cnt+=1


#Fitting logistic regression model over the training data set (english)
log_reg = linear_model.LogisticRegression(C=L2_REGULARIZATION_STRENGTH, penalty='l2', n_jobs=4)
log_reg.fit(Xtrain[:],Y[:])

#Predicting class labels for dataset: chinese
YtestCh = log_reg.predict(Xtester[:])

#Comparison model for english and chinese results:
if len(YtestEn)!=len(YtestCh):
    print('Not same length')
for i in range(len(YtestEn)):
    if YtestEn[i]!=YtestCh[i]:
        print(1)
    else:
        print(0)
