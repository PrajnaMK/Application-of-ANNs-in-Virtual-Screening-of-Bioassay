# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:58:36 2020

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


df1 = pd.read_csv('AID_492953_datatable_all.csv')
df2 = pd.read_csv('AID_492956_datatable_all.csv')
df3= pd.read_csv('AID_492972_datatable_all.csv')

df1 = df1.append(df2, ignore_index =True)
df1 = df1.append(df3, ignore_index =True)


del df1['PUBCHEM_RESULT_TAG']
del df1['PUBCHEM_SID']
del df1['PUBCHEM_ACTIVITY_URL']
del df1['PUBCHEM_ASSAYDATA_COMMENT']




df1=df1.replace({'Inactive':0, 'Active':1})



X=df1.iloc[:,[0,2,3]].values
Y= df1.iloc[:,1].values

X=X.astype(int)

#splitting data and feature scalling
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20)
scaler = StandardScaler()
X_test= scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)


 
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


