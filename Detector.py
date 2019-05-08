#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:01:13 2018

@author: uless
"""
#%%
#Importing packages
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import log, sqrt
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import numpy as np
import glob
import os
from collections import Counter
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
#%%
#Read file & split dataset
file = r'/home/uless/NewDisk/Detector/ML0.xls'
entries = pd.read_excel(file)
print('There are {} entries:'.format(len(entries)))
train, test = train_test_split(entries, test_size=0.3)
print(train['exclusion'].value_counts(),'\n',test['exclusion'].value_counts())
#%%
#Setting stopwords. 
#Note: Shall we also exlude English stopwords?
from stop_words import get_stop_words
stopwords1 = get_stop_words('dutch')
sw=stopwords1#+stopwords.words('english')
#%%
#Setting up vectorizer
count_vectorizer = CountVectorizer(stop_words=sw)
tfidf_vectorizer = TfidfVectorizer(stop_words=sw)
train_features_count = count_vectorizer.fit_transform([e for e in train['text_processed']])
test_features_count = count_vectorizer.transform([e for e in test['text_processed']])
train_features_tfidf = tfidf_vectorizer.fit_transform([e for e in train['text_processed']])
test_features_tfidf = tfidf_vectorizer.transform([e for e in test['text_processed']])
#%%
train_scores = [e for e in train['exclusion']]
actual_scores = [e for e in test['exclusion']]
#%%
def performance(predictions):
    accuracy=metrics.accuracy_score(actual_scores,predictions,normalize=True)
    recall=metrics.recall_score(actual_scores,predictions,pos_label=1,labels=['0','1'])
    recall_incl=metrics.recall_score(actual_scores,predictions,pos_label=0,labels=['0','1'])
    precision=metrics.precision_score(actual_scores,predictions,pos_label=1,labels=['0','1'])
    precision_incl=metrics.precision_score(actual_scores,predictions,pos_label=0,labels=['1','0'])
    confusionmatrix=metrics.confusion_matrix(actual_scores,predictions)
    print('Accuracy:',accuracy,
          '\nRecall:',recall,
          '\nNote: Recall if we are interested in predicting inclusion of entries:',recall_incl,
          '\nPrecision:',precision,
          '\nNote: Precision if we are interested in predicting inclusion of entries:',precision_incl,
          '\nConfusion Matrix:','\n',confusionmatrix)
#%%
#Count_NB
#train & show performance
m1_nb = MultinomialNB()
m1_nb.fit(train_features_count, train_scores)
m1_predictions = m1_nb.predict(test_features_count)
performance(m1_predictions)
#%%
#Save
pk.dump(count_vectorizer,open('count_vectorizer.pkl',mode='wb'))
joblib.dump(m1_nb,'nb_classifier_count.pkl')
#%%
#tf-idf_NB
m2_nb = MultinomialNB()
m2_nb.fit(train_features_tfidf, train_scores)
m2_predictions = m2_nb.predict(test_features_tfidf)
performance(m2_predictions)
#%%
pk.dump(tfidf_vectorizer,open('tfidf_vectorizer.pkl',mode='wb'))
joblib.dump(m2_nb,'nb_classifier_tfidf.pkl')
#%%
#Count_LR
m3_logreg = LogisticRegression()
m3_logreg.fit(train_features_count, train_scores)
m3_predictions = m3_logreg.predict(test_features_count)
performance(m3_predictions)
#%%
joblib.dump(m3_logreg, 'logreg_classifier.pkl')
#%%
#tf-idf_LR
m4_logreg = LogisticRegression()
m4_logreg.fit(train_features_tfidf, train_scores)
m4_predictions = m4_logreg.predict(test_features_tfidf)
performance(m4_predictions)
#%%
joblib.dump(m4_logreg, 'logreg_classifier_tfidf.pkl')
#%%
#make ROCs
m1_predict_probabilities = m1_nb.predict_proba(test_features_count)
fpr, tpr, thresholds = metrics.roc_curve(actual_scores, m1_predict_probabilities[:,1],pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC:\n{}".format(roc_auc))
#%%
m2_predict_probabilities = m2_nb.predict_proba(test_features_tfidf)
fpr, tpr, thresholds = metrics.roc_curve(actual_scores, m2_predict_probabilities[:,1],pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC:\n{}".format(roc_auc))
#%%
m3_predict_probabilities = m3_logreg.predict_proba(test_features_count)
fpr, tpr, thresholds = metrics.roc_curve(actual_scores, m3_predict_probabilities[:,1],pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC:\n{}".format(roc_auc))
#%%
m4_predict_probabilities = m4_logreg.predict_proba(test_features_tfidf)
fpr, tpr, thresholds = metrics.roc_curve(actual_scores, m4_predict_probabilities[:,1],pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC:\n{}".format(roc_auc))
#%%
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#%%
##Create two Lists with TP's and FP's
#%%
predictions = pd.DataFrame(m3_predictions)
predictions.rename(columns={0: 'predictions'}, inplace=True)
assert len(test) == len(predictions)
test = test.reset_index()
df_comparison = test.join(predictions, how = 'outer')
#%%
#clean the text
TP = df_comparison.loc[(df_comparison['predictions'] == 1) & (df_comparison['exclusion'] == 1)]
TP_text1 = list(TP['text_processed'])
TP_text=[]
for i in TP_text1:
    a="".join((char for char in i if char not in string.punctuation))
    TP_text.append(a)
assert len(TP) == len(TP_text)
FP = df_comparison.loc[(df_comparison['predictions'] == 1) & (df_comparison['exclusion'] == 0)]
FP_text1 = list(FP['text_processed'])
FP_text=[]
for i in FP_text1:
    a="".join((char for char in i if char not in string.punctuation))
    FP_text.append(a)
assert len(FP) == len(FP_text)
TP_fulltext1 = " ".join(TP_text)
TP_fulltext = " ".join((word for word in TP_fulltext1.lower().split() if word not in sw))
FP_fulltext1 = " ".join(FP_text)
FP_fulltext = " ".join((word for word in FP_fulltext1.lower().split() if word not in sw))
#%%
#make corpus
corpus_tp = Counter(TP_fulltext.split())
corpus_fp = Counter(FP_fulltext.split())
#%%
##Create a File with Log Likelihood
#%%
def llcompare(corpus1, corpus2, llbestand, llbestand2):
    c = sum(corpus1.values())
    d = sum(corpus2.values())
    ll = {}
    e1dict = {}
    e2dict = {}
    for word in corpus1:
        a = corpus1[word]
        try:
            b = corpus2[word]
        except KeyError:
            b = 0
        e1 = c * (a + b) / (c + d)
        e2 = d * (a + b) / (c + d)
        if a == 0:
            part1 = 0
        else:
            part1 = a * log(a / e1)
        if b == 0:
            part2 = 0
        else:
            part2 = b * log(b / e2)
        llvalue = 2 * (part1 + part2)
        ll[word] = llvalue
        e1dict[word] = e1
        e2dict[word] = e2
        
    for word in corpus2:
        if word not in corpus1:
            a = 0
            b = corpus2[word]
            e2 = d * (a + b) / (c + d)
            llvalue = 2 * (b * log(b / e2))
            ll[word] = llvalue
            e1 = c * (a + b) / (c + d)
            e1dict[word] = e1
            e2dict[word] = e2
    print("Writing results...")
    with open(llbestand+".csv", mode='w', encoding="utf-8") as f, open(llbestand2+".txt", mode='w', encoding="utf-8") as f2:
        f.write("ll,word,freqcorp1,expectedcorp1,freqcorp2,expectedcorp2\n")
        for word, value in ((k, ll[k]) for k in sorted(ll, key=ll.get,reverse=True)):
            try:
                freqcorp1 = corpus1[word]
            except KeyError:
                freqcorp1 = 0
            try:
                freqcorp2 = corpus2[word]
            except KeyError:
                freqcorp2 = 0
            e1 = str(e1dict[word])
            e2 = str(e2dict[word])
            f.write(str(value) + "," + word + "," + str(freqcorp1) +"," + e1 + "," + str(freqcorp2) + "," + e2 + "\n")
            if freqcorp1 > e1dict[word]: 
                f2.write(word + "\n")
    print("Output written to", llbestand+".csv")
    print("Those words whith observed frequency > expected frequency in Corpus 1 are additionally (sorted by descending LL) written to",llbestand2+".txt")
#%%
def llfreqplot(dataleft,dataright, XMARGIN=40, YMARGIN=.2, BASEFONTSIZE=40,YSTRETCH=3):
    dataleft=[(-e[0],e[1],e[2]) for e in dataleft]
    
    dataleft.sort(key=lambda tup: abs(tup[0]))
    dataright.sort(key=lambda tup: abs(tup[0]))
    
    maxll=max([e[0] for e in dataright])
    minll=min([e[0] for e in dataleft])
    maxfreqleft=sqrt(max([e[1] for e in dataleft]))
    maxfreqright=sqrt(max([e[1] for e in dataright]))
    noterms=max([len(dataleft),len(dataright)])
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('Loglikelihood')
    ax.set_ylim(0,YSTRETCH)
    
    i=0
    for e in dataleft:
        i+=1
        if addmarker(e[2]):
            bboxspec=dict(facecolor='yellow', alpha=0.5)
        else:
            bboxspec={}
        ax.text(e[0], i*YSTRETCH, e[2], horizontalalignment='right',fontsize=BASEFONTSIZE * sqrt(e[1])/maxfreqleft,color='b', bbox=bboxspec)
        
    i=0
    for e in dataright:
        i+=1
        if addmarker(e[2]):
            bboxspec=dict(facecolor='yellow', alpha=0.5)
        else:
            bboxspec={}
        ax.text(e[0], i*YSTRETCH, e[2],fontsize=BASEFONTSIZE * sqrt(e[1])/maxfreqright,color='r', bbox=bboxspec)
    
    ax.axis([minll-XMARGIN, maxll+XMARGIN, 1-YMARGIN, noterms+YMARGIN])
# removing negative sign for left side of graph
    labels = [str(int(abs(item))) for item in ax.get_xticks()]
    ax.set_xticklabels(labels)
# removing the weird black box
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
# removing meaningless y axis
    ax.axes.get_yaxis().set_visible(False)
# but add line at y=0:
#ax.axvline(0, color='black',ymax=noterms)
    plt.show()
#%%
def addmarker(word):
#WNTBM='/Users/damian/SURFdrive/onderzoeksprojecten_lopend/2015damianmark_2ndscreen/output/substantivewords-usdebates.txt'
#wordsthatneedtobemarked=[w.strip().lstrip('#') for w in open(WNTBM).readlines()]
    wordsthatneedtobemarked=['Nederland','vaccinaties','vaxxed','Vaccinatieschade'] ##Need a dict for that
    if word.strip().lstrip('#') in wordsthatneedtobemarked:
        return True
    else:
        return False
#%%
def readfromlloutputfile(filename):
    import csv
    dataleft=[]
    dataright=[]
    with open(filename, encoding='utf-8-sig',mode='r',newline='',errors='ignore') as fi:
        reader = csv.reader(fi, delimiter=',')
        next(reader, None) # skip the header row
        for row in reader:
            if int(row[2])>float(row[3]):
                dataleft.append((float(row[0]),int(float(row[3])),row[1].replace('tag_','#')))
            elif int(row[4])>float(row[5]):
                dataright.append((float(row[0]),int(row[4]),row[1].replace('tag_','#')))
    return(dataleft,dataright)
#%%
#Finding features of TP and FP in the test set
llcompare(corpus_tp, corpus_fp, 'll_full', 'll_overrepresented')
#%%
dataleft, dataright = readfromlloutputfile('ll_full.csv')
#%%
dataleft[:10]
#%%
llfreqplot(dataleft[:10],dataright[:10], XMARGIN=40, YMARGIN=.2, BASEFONTSIZE=40,YSTRETCH=2)
#%%
#Finding features of Misinfo/Correctinfo in the training set
opsplitsen = train.groupby('exclusion').agg(lambda x:' '.join(x))
#%%
#also cleaning the text
nex1="".join((char for char in opsplitsen.iloc[0].to_dict()['text_processed'] if char not in string.punctuation))
wex1="".join((char for char in opsplitsen.iloc[1].to_dict()['text_processed'] if char not in string.punctuation))
nex = " ".join((word for word in nex1.lower().split() if word not in sw))
wex =" ".join((word for word in wex1.lower().split() if word not in sw))
#%%
nietexcluded = Counter(nex.split())
welexcluded = Counter(wex.split())
llcompare(nietexcluded, welexcluded, 'll_full_aldannietexcluded', 'll_overrepresented')
#%%
dataleft, dataright = readfromlloutputfile('ll_full_aldannietexcluded.csv')
#%%
llfreqplot(dataleft[:10],dataright[:10], XMARGIN=40, YMARGIN=.2, BASEFONTSIZE=40,YSTRETCH=2)
