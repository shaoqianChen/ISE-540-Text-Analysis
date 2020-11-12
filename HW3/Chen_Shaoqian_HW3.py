#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:28:51 2020

@author: shaoqianchen
"""
import math
import numpy as np
import re
import os
import random
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import bernoulli

train_path = "/Users/shaoqianchen/Desktop/2020 Fall/ISE 540/HW3/files/train"
test_path = "/Users/shaoqianchen/Desktop/2020 Fall/ISE 540/HW3/files/test" 


def split_subject_main(file):
    #################################################
    # Split a txt file #
    #################################################
    # split a given txt file into two parts: subject and main
    open_file = open(file)
    line = open_file.read().replace("\n", " ")
    open_file.close()
    split_list = re.split(r'\s{2,}', line)
    # Remove string 'Subject:' from subject part
    try:
        split_list[0] = split_list[0].replace('Subject:','')
    except:
        print("No subject part in the given file")
    #retrns a list contains 2 strings: subject and main
    # [subject,main]
    return split_list

"""
[30 points] Replace 'pass' below as appropriate.

"""
def encode_text_tfidf_vectors(in_folder, subject):
    folder_dirs = os.listdir(in_folder) #list of all filenames in in_folder
    y = np.array([])
    # 0 represent spam, 1 represent legit
    for i in folder_dirs:
        if 'legit' in i:
            y = np.insert(y,0,1)
        elif 'spmsg' in i:
            y = np.insert(y,0,0)
    corpus_subject = []        
    corpus_main = []
    vectorizer = TfidfVectorizer(ngram_range=(2,2),sublinear_tf=True)
    for file in folder_dirs:
        if subject == True: # subject only
            corpus_subject.append(split_subject_main(in_folder+"/"+file)[0])
            x = vectorizer.fit_transform(corpus_subject)
            X = x.toarray()
        elif subject == False: # main only
            corpus_main.append(split_subject_main(os.path.abspath(in_folder+"/"+file))[1])
            x = vectorizer.fit_transform(corpus_main)
            X = x.toarray()
    return X, y




"""
[30 points] Train a classifier on the train dataset that is able to predict spam. 
""" 
def training_testing_split(X, y, training_ratio): 
    #####################
    #Stratified Sampling#
    #####################
    #code from my HW2 
    # convert input data into list
    if type(X)==np.ndarray:
        X = X.tolist()
    if type(y)==np.ndarray:
        y = y.tolist() 
    X_pos = []
    X_neg = []
    #split 1 and 0 instances 
    for i in range(len(y)):
        if y[i] == 1 or y[i]==[1]: 
            X_pos.append(X[i])
        elif y[i] == 0 or y[i]==[0]: 
            X_neg.append(X[i])
        
    X_train_pos_sample_index = random.sample(range(len(X_pos)),math.ceil(len(X_pos)*training_ratio))#index
    X_train_neg_sample_index = random.sample(range(len(X_neg)),math.ceil(len(X_neg)*training_ratio))#index
    X_test_pos_sample_index=[] 
    X_test_neg_sample_index=[] #fill all X_test indexes 
    
    for j in range(len(X_pos)):
        if j not in X_train_pos_sample_index: 
            X_test_pos_sample_index.append(j)
    for k in range(len(X_neg)):
        if k not in X_train_neg_sample_index:
            X_test_neg_sample_index.append(k)
    X_train=[]
    X_test=[]
    for m in X_train_pos_sample_index:
        X_train.append(X[m])
    for n in X_train_neg_sample_index:
        X_train.append(X[n])
    for o in X_test_pos_sample_index:
        X_test.append(X[o])
    for p in X_test_neg_sample_index:
        X_test.append(X[p])
    y_train = np.array([0]*len(X_train_neg_sample_index)+[1]*len(X_train_pos_sample_index))
    y_test = np.array([0]*len(X_test_neg_sample_index)+[1]*len(X_test_pos_sample_index))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    #convert all output to nparray
    return (X_train, y_train, X_test, y_test)

def fit_classifier(classifier,X_train,y_train,X_test,y_test):
    # Fit classifiers 
    # Return accuracy_score and average cross_validation_score
    if classifier == "RandomForest":
        clf = RandomForestClassifier(max_depth=2, random_state=0)
    elif classifier == "GaussianNB":     
        clf = GaussianNB()
    elif classifier == "KNeighbors":
        clf = KNeighborsClassifier(n_neighbors=3)
    elif classifier == "SVC":
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    elif classifier == "MLP":
        clf = MLPClassifier(random_state=1, max_iter=300)
        
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5)
    return (accuracy,all_accuracies.mean())

def compare_classifiers(classifiers,X_train, y_train, X_test, y_test):
    #function to print out accuracy and 5-CV_score for all listed classifiers
    for clfs in classifiers:    
        print(fit_classifier(clfs, X_train, y_train, X_test, y_test))

"""Matrix = encode_text_tfidf_vectors(train_path, False)
M = Matrix[0]
n = Matrix[1]

split = training_testing_split(M,n,0.8)
X_train = split[0]
X_test = split[2]
y_train = split[1]
y_test = split[3]    
classifiers = ["MLP", "SVC", "KNeighbors", "GaussianNB", "RandomForest"]
compare_classifiers(classifiers)"""


""" [10 points] Using the best and second-best classifiers you got from above, 
apply them once each on the entire test dataset"""

def get_all_corpus(in_folder, subject):
    folder_dirs = os.listdir(in_folder) #list of all filenames in in_folder
    y = np.array([])
    # 0 represent spam, 1 represent legit
    for i in folder_dirs:
        if 'legit' in i:
            y = np.insert(y,0,1)
        elif 'spmsg' in i:
            y = np.insert(y,0,0)
        
    corpus = []        
    for file in folder_dirs:
        if subject == True: # subject only
            corpus.append(split_subject_main(in_folder+"/"+file)[0])
        elif subject == False: # main only
            corpus.append(split_subject_main(os.path.abspath(in_folder+"/"+file))[1])

    return corpus

#A function to return all unique elements in a corpus
def corpus_toset(corpus):
    full_set_int = set()
    for i in corpus:
        a_list = i.split()
        map_object = map(int,a_list)
        set_of_int = set(list(map_object))
        full_set_int |= set_of_int
    return full_set_int


def encode_corpus_tfidf_vector(corpus):
    vectorizer = TfidfVectorizer(ngram_range=(2,2),max_df=1.0, min_df=1, use_idf=True, smooth_idf=True, sublinear_tf=True)
    x = vectorizer.fit_transform(corpus)
    X = x.toarray()   
    return X


#function to remove corpus in a but not in b and return new_a
def remove_a_notin_b_return_A(a,b):
    A = []
    for ele in a:
        temp_cop = ele.split()
        temp_aa = ele
        for i in range(len(temp_cop)):
            if int(temp_cop[i]) not in corpus_toset(b):
                temp_aa=temp_aa.replace(temp_cop[i],"")
                print(temp_cop[i])
        A.append(temp_aa)
    return A

############ 
#Question 2_1,2_2#
############
"""Matrix_train = encode_text_tfidf_vectors(train_path, False)
Matrix_test = encode_text_tfidf_vectors(test_path, False)
 
#X_train = Matrix_train[0]
y_train = Matrix_train[1]

#X_test = Matrix_test[0]
y_test = Matrix_test[1]"""

"""
train_corpus = get_all_corpus(train_path, True) 
test_corpus = get_all_corpus(test_path, True)

train_corpus_cleaned = remove_a_notin_b_return_A(train_corpus, test_corpus)
test_corpus_cleaned = remove_a_notin_b_return_A(test_corpus, train_corpus_cleaned)

 
X_test = encode_corpus_tfidf_vector(test_corpus_cleaned)
X_train = encode_corpus_tfidf_vector(train_corpus_cleaned)
 
#top_class = ['RandomForest','SVC']
#compare_classifiers(top_class,X_train, y_train, X_test, y_test)"""
 
"""
Matrix = encode_text_tfidf_vectors(test_path, False)
M = Matrix[0]
n = Matrix[1]

split = training_testing_split(M,n,0.8)
X_train = split[0]
X_test = split[2]
y_train = split[1]
y_test = split[3]    
classifiers = [ "SVC","RandomForest"]
compare_classifiers(classifiers,X_train, y_train, X_test, y_test)"""
 
"""Output
(0.8461538461538461, 0.7939393939393941)
(0.8461538461538461, 0.7939393939393941)
 """

############ 
#Question 2_3#
############
#find p using training data
"""Matrix = encode_text_tfidf_vectors(train_path, False)
y = Matrix[1]
def count_ins_y(y): 
    pos = 0
    neg = 0 
    for i in y:
        if i==0 or i==[0]: 
            neg+=1
        if i==1 or i == [1]: 
            pos+=1
    p = neg/pos
    print("Number of 1: ",pos) 
    print("Number of 0: ",neg)
    print("p = ", p)
    return (pos,neg,p)
p = count_ins_y(y)[2]

#p =  0.24561403508771928
Matrix = encode_text_tfidf_vectors(test_path, False)
M = Matrix[0]
n = Matrix[1]

split = training_testing_split(M,n,0.8)
X_train = split[0]
X_test = split[2]
y_train = split[1]
y_test = split[3]

y_train = bernoulli.rvs(1-p, size=len(y_train))

classifiers = [ "SVC","RandomForest"]
compare_classifiers(classifiers,X_train, y_train, X_test, y_test)"""

"""Output
(0.6153846153846154, 0.7590909090909091)
(0.8461538461538461, 0.7939393939393941)
"""
 
 
############## 
#Extra Credit#
############## 


#Test Dataset subject analysis
"""Matrix = encode_text_tfidf_vectors(test_path, True)
M = Matrix[0]
n = Matrix[1]
split = training_testing_split(M,n,0.8)
X_train = split[0]
X_test = split[2]
y_train = split[1]
y_test = split[3]    
classifiers = ["MLP", "SVC", "KNeighbors", "GaussianNB", "RandomForest"]
compare_classifiers(classifiers,X_train, y_train, X_test, y_test)"""

""" Output (Accuracy,CV_score)
(0.6923076923076923, 0.7590909090909091)
(0.7692307692307693, 0.7772727272727273)
(0.8461538461538461, 0.7939393939393941)
(0.23076923076923078, 0.5015151515151516)
(0.8461538461538461, 0.7772727272727273)
"""







 
 
 
 
 
 
