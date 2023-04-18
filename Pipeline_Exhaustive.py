import csv
import numpy as np
import pandas as pd
import re
import time


alex_layers = 8
gg_layers = 13
res_layers = 18
mobi_layers = 15
squ_layers = 10

file="pred"
pd_latency = pd.read_csv('schedule/'+file+'/'+'res_latency1.csv')
model_layers = mobi_layers

pipeline = [
    ["gpu", "1b", "1b", "1b", "1b", "1s", "1s", "1s", "1s"],
    ["gpu", "1b", "1b", "1b", "1b", "2s", "1s", "1s"],
    ["gpu", "1b", "1b", "1b", "1b", "1s", "2s", "1s"],
    ["gpu", "1b", "1b", "1b", "1b", "1s", "1s", "2s"],
    ["gpu", "1b", "1b", "1b", "1b", "2s", "2s"],
    ["gpu", "1b", "1b", "1b", "1b", "3s", "1s"],
    ["gpu", "1b", "1b", "1b", "1b", "1s", "3s"],
    ["gpu", "1b", "1b", "1b", "1b", "4s"],

    ["gpu", "2b", "1b", "1b", "1s", "1s", "1s", "1s"],
    ["gpu", "2b", "1b", "1b", "2s", "1s", "1s"],
    ["gpu", "2b", "1b", "1b", "1s", "2s", "1s"],
    ["gpu", "2b", "1b", "1b", "1s", "1s", "2s"],
    ["gpu", "2b", "1b", "1b", "2s", "2s"],
    ["gpu", "2b", "1b", "1b", "3s", "1s"],
    ["gpu", "2b", "1b", "1b", "1s", "3s"],
    ["gpu", "2b", "1b", "1b", "4s"],


    ["gpu", "1b", "2b", "1b", "1s", "1s", "1s", "1s"],
    ["gpu", "1b", "2b", "1b", "2s", "1s", "1s"],
    ["gpu", "1b", "2b", "1b", "1s", "2s", "1s"],
    ["gpu", "1b", "2b", "1b", "1s", "1s", "2s"],
    ["gpu", "1b", "2b", "1b", "2s", "2s"],
    ["gpu", "1b", "2b", "1b", "3s", "1s"],
    ["gpu", "1b", "2b", "1b", "1s", "3s"],
    ["gpu", "1b", "2b", "1b", "4s"],


    ["gpu", "1b", "1b", "2b", "1s", "1s", "1s", "1s"],
    ["gpu", "1b", "1b", "2b", "2s", "1s", "1s"],
    ["gpu", "1b", "1b", "2b", "1s", "2s", "1s"],
    ["gpu", "1b", "1b", "2b", "1s", "1s", "2s"],
    ["gpu", "1b", "1b", "2b", "2s", "2s"],
    ["gpu", "1b", "1b", "2b", "3s", "1s"],
    ["gpu", "1b", "1b", "2b", "1s", "3s"],
    ["gpu", "1b", "1b", "2b", "4s"],


    ["gpu", "2b", "2b", "1s", "1s", "1s", "1s"],
    ["gpu", "2b", "2b", "2s", "1s", "1s"],
    ["gpu", "2b", "2b", "1s", "2s", "1s"],
    ["gpu", "2b", "2b", "1s", "1s", "2s"],
    ["gpu", "2b", "2b", "2s", "2s"],
    ["gpu", "2b", "2b", "3s", "1s"],
    ["gpu", "2b", "2b", "1s", "3s"],
    ["gpu", "2b", "2b", "4s"],


    ["gpu", "3b", "1b", "1s", "1s", "1s", "1s"],
    ["gpu", "3b", "1b", "2s", "1s", "1s"],
    ["gpu", "3b", "1b", "1s", "2s", "1s"],
    ["gpu", "3b", "1b", "1s", "1s", "2s"],
    ["gpu", "3b", "1b", "2s", "2s"],
    ["gpu", "3b", "1b", "3s", "1s"],
    ["gpu", "3b", "1b", "1s", "3s"],
    ["gpu", "3b", "1b", "4s"],


    ["gpu", "1b", "3b", "1s", "1s", "1s", "1s"],
    ["gpu", "1b", "3b", "2s", "1s", "1s"],
    ["gpu", "1b", "3b", "1s", "2s", "1s"],
    ["gpu", "1b", "3b", "1s", "1s", "2s"],
    ["gpu", "1b", "3b", "2s", "2s"],
    ["gpu", "1b", "3b", "3s", "1s"],
    ["gpu", "1b", "3b", "1s", "3s"],
    ["gpu", "1b", "3b", "4s"],

    ["gpu", "4b", "1s", "1s", "1s", "1s"],
    ["gpu", "4b", "2s", "1s", "1s"],
    ["gpu", "4b", "1s", "2s", "1s"],
    ["gpu", "4b", "1s", "1s", "2s"],
    ["gpu", "4b", "2s", "2s"],
    ["gpu", "4b", "3s", "1s"],
    ["gpu", "4b", "1s", "3s"],
    ["gpu", "4b", "4s"]
]

def num_combinations(start, end):
    list = []
    if (start == end - 1):
        ll = []
        ll.append(start)
        list.append(ll)
        return list
    else:
        for i in range(start, end):
            # print("i= ", i)
            if(i == end - 1):
                list.append([i])
                # print("listt= ", list)
            else:
                tail_list = num_combinations(i+1, end)
                # print("tail_list= ", tail_list)
                for j in range(len(tail_list)):
                    l = []
                    l.append(i)
                    l.extend(tail_list[j])
                    list.append(l)
                    # print("list= ", list)
        return list

def cal_time(L, P):
    stage_time_list = []
    first_stage_time = 0
    for i in range(0, L[0]+1):
        first_stage_time = first_stage_time + pd_latency[P[0]][i]
    stage_time_list.append(first_stage_time)
    for j in range(1, len(L)):
        t = 0
        for k in range(L[j-1]+1, L[j]+1):
            t = t + pd_latency[P[j]][k]
        stage_time_list.append(t)
    return stage_time_list

def all_time(list, num_layers):
    best_pipeline = []
    best_combination = []
    best_stage_time = []
    min_maxtime = 10000
    min_E = 10000
    for i in range(len(pipeline)):
        P = pipeline[i]
        P_len = len(P)
        if (P_len < num_layers + 1):
            L_all = list[P_len]
            for j in range(len(L_all)):
                L = L_all[j]
                T = cal_time(L, P)
                print("L: ", L)
                print("P: ", P)
                print("T: ", T)
                sum_time = 0
                for m in range(len(T)):
                    sum_time = sum_time + T[m]
                avg_time = sum_time / len(T)
                print("avg_time: ", avg_time)
                E = 0
                for n in range(len(T)):
                    E = E + (avg_time - T[n])**2
                E = E / len(T)
                print("max_stage_time= ", max(T))
                print("方差为：         ", E)
                if(min_maxtime > max(T)):
                    min_maxtime = max(T)
                    min_E = E
                    best_pipeline = P
                    best_stage_time = T
                    best_combination = L
                elif(min_maxtime == max(T) and min_E > E):
                    min_maxtime = max(T)
                    min_E = E
                    best_pipeline = P
                    best_stage_time = T
                    best_combination = L
                print("\n")
    print("best_pipeline: ", best_pipeline)
    print("best_combination: ", best_combination)
    print("best_stage_time: ", best_stage_time)
    print("max_time: ", min_maxtime)
    print("min_E: ", min_E)


start = time.perf_counter()

a = num_combinations(0, model_layers)
print("total combinations: ", len(a))
print("list: ", a)
count = 0
useful_list = []
for i in range(0, 10):
    useful_list.append([])
for k in range(len(a)):
    combinations = a[k]
    if(len(a[k]) < 10 and len(a[k]) > 2):
        count = count + 1
        useful_list[len(a[k])].append(a[k])
print("useful combinations: ", count)
print("useful list: ", useful_list)
all_time(useful_list, model_layers)

end = time.perf_counter()
print ("***************\n", end-start)




