# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:04:20 2020

@author: prajn
"""


from __future__ import division
import csv
from random import randrange
import numpy as np
import urllib
import math
import random
import urllib.request
from random import randrange
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


df1 = pd.read_csv('AID_686971_datatable_all.csv')
df2 = pd.read_csv('AID_686970_datatable_all.csv' )
df1=df1.append(df2, ignore_index = True)

print(df1.shape)
df1.describe().transpose()

df1=df1.replace({'Inactive':0, 'Active':1, 'Inconclusive':2, 'Cytotoxic':3})


X=df1.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13]].values
Y= df1.iloc[:,1].values


X=X.astype(int)
Y=Y.astype(int)

#splitting data and feature scalling
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20)
scaler = StandardScaler()
X_test= scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)

#applying dimension reduction/PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= 3)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
var_explained = pca.explained_variance_ratio_
'''
#applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=3)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)
'''

mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=200, activation='relu', solver='sgd')
mlp.fit(X_train,Y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
cm1=confusion_matrix(Y_train, predict_train)
cm2= confusion_matrix(Y_test, predict_test)



import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(Y_test, predict_test)



from sklearn import metrics
accuracy = metrics.r2_score(Y_test, predict_test)
print ("accuracy :", accuracy*100,"%")

