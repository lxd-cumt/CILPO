# -*- coding: utf-8 -*
from numpy import *
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #这里是引用了交叉验证
from sklearn.linear_model import LinearRegression
import os
from sklearn.preprocessing import StandardScaler


file=['ht-cpu-s1.csv','ht-cpu-s2.csv','ht-cpu-s3.csv','ht-cpu-s4.csv','ht-cpu-b1.csv','ht-cpu-b2.csv','ht-cpu-b3.csv','ht-cpu-b4.csv']
for f in file:
    pd_data = pd.read_csv('dataset-train/'+f,header=-1)
    pd_data1 = pd.read_csv('modelfeature/mobi.csv',header=-1)

    x_train = pd_data.loc[:, 0:6]
    y_train = pd_data.loc[:, 7]
    x_test = pd_data1.loc[:, 0:6]

    # ss_x = StandardScaler()
    # x_train = ss_x.fit_transform(x_train)
    # x_test = ss_x.transform(x_test)

    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)

    y_pred = linreg.predict(x_test)

    print sum(y_pred)
