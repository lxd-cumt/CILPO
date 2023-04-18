# -*- coding: utf-8 -*
from numpy import *
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import os
from sklearn.preprocessing import StandardScaler

# file = ['gpu_train_dw_layer.csv', 'gpu_train_dw_fill.csv', 'gpu_train_dw_run.csv']

# file=['cpu-s1-dw.csv','cpu-s2-dw.csv','cpu-s3-dw.csv','cpu-s4-dw.csv',
#     'cpu-b1-dw.csv','cpu-b2-dw.csv','cpu-b3-dw.csv','cpu-b4-dw.csv']


for f in file:
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)

    pd_data1 = pd.read_csv('modelfeature/mobi-dw.csv',header=-1)


    x_train = pd_data.loc[:, 0:2]
    y_train = pd_data.loc[:, 3]
    x_test = pd_data1.loc[:, 0:2]

    # ss_x = StandardScaler()
    # x_train = ss_x.fit_transform(x_train)
    # x_test = ss_x.transform(x_test)

    # linear
    # linreg = LinearRegression()
    # model = linreg.fit(x_train, y_train)
    # y_pred = linreg.predict(x_test)
    # print sum(y_pred)

    # DecisionTree
    classifier = DecisionTreeRegressor()
    classifier.fit(x_train,y_train)
    y_predict=classifier.predict(x_test)
    # print sum(y_predict), y_predict.tolist()
    print sum(y_predict)
    # print
    # print ()
    # for ch in y_predict:
    #     print ch,
    # print