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


# file=['cpu-s1-im2col.csv','cpu-s2-im2col.csv','cpu-s3-im2col.csv','cpu-s4-im2col.csv',
#     'cpu-b1-im2col.csv','cpu-b2-im2col.csv','cpu-b3-im2col.csv','cpu-b4-im2col.csv']

# file=['cpu-s1-col2im.csv','cpu-s2-col2im.csv','cpu-s3-col2im.csv','cpu-s4-col2im.csv',
#     'cpu-b1-col2im.csv','cpu-b2-col2im.csv','cpu-b3-col2im.csv','cpu-b4-col2im.csv']

file=['cpu-s1-gemmrun.csv','cpu-s2-gemmrun.csv','cpu-s3-gemmrun.csv','cpu-s4-gemmrun.csv',
    'cpu-b1-gemmrun.csv','cpu-b2-gemmrun.csv','cpu-b3-gemmrun.csv','cpu-b4-gemmrun.csv']

# file=['cpu-s1-fc.csv','cpu-s2-fc.csv','cpu-s3-fc.csv','cpu-s4-fc.csv',
#     'cpu-b1-fc.csv','cpu-b2-fc.csv','cpu-b3-fc.csv','cpu-b4-fc.csv']


for f in file:
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)
    # pd_data1 = pd.read_csv('modelfeature/kernel/squ.csv',header=-1)
    pd_data1 = pd.read_csv('modelfeature/mobi-conv.csv',header=-1)

    # x_train = pd_data.loc[:, 0:5]
    # y_train = pd_data.loc[:, 6]
    # x_test = pd_data1.loc[:, 0:5]

    x_train = pd_data.loc[:, 0:8]
    y_train = pd_data.loc[:, 9]
    x_test = pd_data1.loc[:, 0:8]

    # fc
    # x_train = pd_data.loc[:, 6:8]
    # y_train = pd_data.loc[:, 9]
    # x_test = pd_data1.loc[:, 0:2]

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
    # print  y_predict.tolist()
    print sum(y_predict.tolist())
    
    # for ch in y_predict:
    #     print ch,
    # print