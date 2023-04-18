# -*- coding: utf-8 -*
from numpy import *
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression

import os


# file=['cpu-s1.csv','cpu-s2.csv','cpu-s3.csv','cpu-s4.csv','cpu-b1.csv','cpu-b2.csv','cpu-b3.csv','cpu-b4.csv']
file=['gpu_dataset.csv']

count = 0
sumerror=0
for f in file:
    # print f
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)
    pd_data1 = pd.read_csv('modelfeature/squ.csv',header=-1)

    x_train = pd_data.loc[:, 0:7]
    y_train = pd_data.loc[:, 8]
    x_test = pd_data1.loc[:, 0:7]


    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    classifier = DecisionTreeRegressor()
    classifier.fit(x_train,y_train)
    y_predict=classifier.predict(x_test)
    print y_predict
    # y_predict = pd.DataFrame({'RISK':y_predict})
    # y_predict.to_csv(f,index=False)
    print sum(y_predict)
