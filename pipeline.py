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
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


L2_REGULARIZATION_STRENGTH = 0.9

headers = {
    'Content-Type': 'application/json',
}

params = (
    ('key', 'AIzaSyCNy3RLPbblytD5Uejh4GkiBb3wAgHENbI'),
)

nlp = spacy.load('en_core_web_sm')
feature_dict = {}
feature_set = {}

def readData(file_name):
    data = pd.read_csv(file_name)
    return data


df = readData('Project_Politeness/Batch_1_Binary_Scores.csv')
df_tags = readData('Project_Politeness/Batch_1_Tags.csv')

dataobjects = df.values.tolist()
tagobjects = df_tags.values.tolist()

cnt=0

for line in dataobjects:
    if cnt == 0:
        cnt=1
        continue
    feature_dict[cnt]={}

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(line[2])
    feature_dict[cnt]['pos']=vs['pos']
    feature_dict[cnt]['neg']=vs['neg']
    feature_dict[cnt]['neu']=vs['neu']

    feature_set['pos']=1
    feature_set['neg']=1
    feature_set['neu']=1
    #print(( (vs['pos'], vs['neg'], vs['neu'])))
    sc = textstat.flesch_reading_ease(line[2])
    feature_dict[cnt]['easiness']=sc
    feature_set['easiness']=1
    #print('Easiness to read', sc)

    stry = str(line[2])
    sent = ''
    for a in stry:
        if a==' ' or (a<='Z' and a>='A') or (a<='z' and a>='a') or (a<='9' and a>='0') or a=='?' or a=='.':
            sent +=a
    data = '{comment: {text:"'+sent+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
    response = requests.post('https://commentanalyzer. .com/v1alpha1/comments:analyze', headers=headers, params=params, data=data)
    j = json.loads(response.text)
    feature_dict[cnt]['toxicity'] =0.0
    try:
        feature_dict[cnt]['toxicity'] = j['attributeScores']['TOXICITY']['summaryScore']['value']
        #print (j['attributeScores']['TOXICITY']['summaryScore']['value'])
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

    doc = nlp(line[2])
    for token in doc:
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        if (str(token.pos_) not in feature_set):
            feature_set[str(token.pos_)]=1

        if not (str(token.pos_) in feature_dict[cnt]):
            feature_dict[cnt][str(token.pos_)]=1
        else:
            feature_dict[cnt][str(token.pos_)]+=1
        #print(str(token.pos_))
    #print(cnt,feature_dict[cnt]['toxicity'])
    cnt+=1

en_col = 34
count=0
for line in tagobjects:
    if count==0:
       count=1
       continue
    cell_count=0
    for cell in line:
        if cell_count<3:
            cell_count+=1
            continue
        if cell_count>en_col:
            break
        cur = ''
        for a in str(cell):
            if (a>='a' and a<='z') or (a>='A' and a <='Z'):
                cur+=a
            if a==',':
                #print(cur)
                feature_dict[count][cur.lower()]=1
                feature_set[cur.lower()]=1
                cur=''

feature_matrix = []
for i in range(1, cnt):
    feature_list = []
    for key in feature_set.keys():
        if key in feature_dict[i]:
            feature_list.append(feature_dict[i][key])
        else:
            feature_list.append(0.0)
    #print(feature_list)
    feature_matrix.append(feature_list)

Y = []
cnt = 0
for line in dataobjects:
    if cnt==0:
        cnt=1
        continue
    Y.append(line[-2])
    cnt+=1
#print(len(feature_matrix))
#print(Y)
X = np.array(feature_matrix)

#log_reg = linear_model.SGDRegressor()
#log_reg = linear_model.BayesianRidge()
#log_reg = linear_model.LassoLars()
#log_reg = linear_model.ARDRegression()
#log_reg = linear_model.PassiveAggressiveRegressor()
#log_reg = linear_model.TheilSenRegressor()
#log_reg = linear_model.LinearRegression()
log_reg = linear_model.LogisticRegression(C=L2_REGULARIZATION_STRENGTH, penalty='l2', n_jobs=4)
log_reg.fit(X[50:250],Y[50:250])
Ytest = log_reg.predict(X[0:50])

#scores = precision_recall_fscore_support(Y[250:], Ytest, average='macro')
#print(scores)
scores = precision_recall_fscore_support(Y[0:50], Ytest, average='micro')
acc = accuracy_score(Y[0:50], Ytest)
print('precision: ',scores[0],' recall: ',scores[1],' f-score: ',scores[2],' accuracy: ',acc)
