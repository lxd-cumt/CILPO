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


# file=['gpu-tensor.csv']  # GPU output tensor
file=['gpu_train_im2col.csv', 'gpu_train_mm.csv', 'gpu_train_reshape.csv', 'gpu_train_col2im.csv', 'gpu_train_layer.csv']
# file=['gpu_train_fc1.csv']

for f in file:
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)

    # pd_data1 = pd.read_csv('modelfeature/fc-alex.csv',header=-1)
    pd_data1 = pd.read_csv('modelfeature/mobi-conv.csv',header=-1)

    # GPU output tensor
    # x_train = pd_data.loc[:, 0:7]
    # y_train = pd_data.loc[:, 8]
    # x_test = pd_data1.loc[:, 0:7]

    # gpu conv ernel
    x_train = pd_data.loc[:, 0:8]
    y_train = pd_data.loc[:, 9]
    x_test = pd_data1.loc[:, 0:8]

    # gpu fc
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
    print sum(y_predict)
    # print y_predict.tolist()
    # for ch in y_predict:
    #     print ch,
    # print