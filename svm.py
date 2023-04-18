import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from numpy import *
import csv
import os

file=['cpu-s1.csv','cpu-s2.csv','cpu-s3.csv','cpu-s4.csv','cpu-b1.csv','cpu-b2.csv','cpu-b3.csv','cpu-b4.csv']

for f in file:
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)
    pd_data1 = pd.read_csv('modelfeature/alex.csv',header=-1)

    x_train = pd_data.loc[:, 0:7]
    y_train = pd_data.loc[:, 8]*1000000000
    x_test = pd_data1.loc[:, 0:7]

    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    # classifier = svm.SVC(C=1.0,kernel='rbf', degree=10, gamma='auto',coef0=0.0,tol=0.001,cache_size=200, max_iter=10)
    classifier = svm.SVC()
    classifier.fit(x_train,y_train.astype('int'))
    y_predict=classifier.predict(x_test)

    print y_predict/1000000000

    # print sum(y_predict)/1000000000
