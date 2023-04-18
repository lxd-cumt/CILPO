# 几种情况 pred-layer + allocation.py ---->论文中的数据
#         pred    + allocation.py  --->>现在可以通过allocation测到的数据
#         pred    + allocation.py(修改过的)-------->ComputeLibrary/examples/after_kernel模型，测不到了
#         pred    + pipeline1.py ------->复现并优化了pipeit的东西，使用的仍然是merge函数，pipeline.txt
#         pred    + pipeline2.py ------------------>暴力穷举流水线分区，仍然使用find_spilt和work_flow函数，现在的分区结果ComputeLibrary/examples/after_kernel_pipeline2模型,pipeline2.txt
#         pred    + pipeline_exhaustive.py   -------->暴力穷举整个解空间
#         证明暴力穷举流水线算法+find_spilt+work_flow优于暴力穷举整个解空间，就是观察：在相同的最优流水线决定策略下，得到的解是否相同或接近
#         并且时间更短。因此两种算法要用相同的最优流水线决定策略。本次实验策略根据pipeline.py决定(和人工判断的最优流水线相比较)，首先选取流水线阶段
#         最大值最小的流水线，然后选取方差最小的流水线划分策略。发现这种策略和人工决定的最优流水线差不多(只有alexnet有细微差别)，因此确定使用了
#         这种最优流水线决定策略，并且两种算法都要使用这种策略。
#
import csv
import numpy as np
import pandas as pd
import re
import time

# Read Predicted Time
file="pred"
pd_latency = pd.read_csv('schedule/'+file+'/'+'squ_latency1.csv')

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

# Find_spilt平衡两个流水线阶段
def find_split(L_wl, P_i, P_i1, TP_i, TP_i1):
    L_i = L_wl[:]
    L_i1 = []
    TP_i_new = TP_i
    TP_i1_new = TP_i1

    for j in range(len(L_wl)):
        layer = L_wl[len(L_wl)-1-j]
        TP_i_new = TP_i_new - pd_latency[P_i][layer]
        TP_i1_new = TP_i1_new + pd_latency[P_i1][layer]
        if(TP_i_new > TP_i1_new):
            L_i.remove(layer)
            L_i1.insert(0, layer)
        else:
            distance = abs(TP_i_new - TP_i1_new)
            TP_i_new = TP_i_new + pd_latency[P_i][layer]
            TP_i1_new = TP_i1_new - pd_latency[P_i1][layer]
            prior_distance = abs(TP_i_new - TP_i1_new)
            if(distance < prior_distance):
                L_i.remove(layer)
                L_i1.insert(0, layer)
            break
    return L_i, L_i1

def Init(p, l_wl):
    num_stages = len(p)
    num_layers = len(l_wl)
    list = []
    for j in range(num_stages):
        list.append([])
    index = 0
    base_layer = 0
    while(num_layers > 0):
        layers = 0
        if(num_layers % num_stages == 0):
            layers = num_layers//num_stages
        else:
            layers = (num_layers//num_stages) + 1
        temp = []
        for j in range(base_layer, base_layer + layers):
            temp.append(j)
        list[index] = temp
        index = index + 1
        base_layer = base_layer + layers
        num_layers = num_layers - layers
        num_stages = num_stages - 1
    return list

# work flow重新划分流水线
def work_flow(P, L_WL):
    # L = []
    # for j in range(len(P)):
    #     L.append([])
    # L[0] = L_WL[:]
    L = Init(P, L_WL)
    # print("work_flow L= ", L)
    L_old = []
    while (L != L_old):
        L_old = L[:]
        for k in range(len(P)-1):
            L_temp = L[k][:]
            L_temp.extend(L[k+1])
            TP_k = 0
            for m in range(len(L_temp)):
                TP_k = TP_k + pd_latency[P[k]][L_temp[m]]
            L[k], L[k+1]= find_split(L_temp, P[k], P[k+1], TP_k, 0)
        # print("work_flow after_split= ", L)
    return L

# 计算流水线每个阶段的时间
def stage_time(L, P):
    T = []
    for j in range(len(L)):
        time = 0
        for k in range(len(L[j])):
            time = time + pd_latency[P[j]][L[j][k]]
        T.append(time)
    return T

# 流水线阶段考虑GPU，merge
def merge_stage(L_WL, HB, HS):
    gpu  = ["gpu"]
    cpub = []
    cpus = []
    P = gpu[:]
    for j in range(HB):
        cpub.append("1b")
    for j in range(HS):
        cpus.append("1s")
    P.extend(cpub)
    P.extend(cpus)
    L = work_flow(P, L_WL)
    num_stages = [0, 1, 5, 9]
    # LOOP: Big Cluster
    j = num_stages[1]
    while(j < num_stages[2]-1):
        # 计算原来两个stage的时间
        t1 = 0
        t2 = 0
        t_max = 0
        for m in range(len(L[j])):
            t1 = t1 + pd_latency[P[j]][L[j][m]]
        for n in range(len(L[j+1])):
            t2 = t2 + pd_latency[P[j+1]][L[j+1][n]]
        t_max = max(t1, t2)
        # 计算stage合并后的时间，确定是否需要合并
        t_after_merge = 0
        change = int(re.findall(r"\d+\.?\d*", P[j])[0]) + int(re.findall(r"\d+\.?\d*", P[j+1])[0])
        str_ = str(change) + "b"
        layer_after_merge = L[j][:]
        layer_after_merge.extend(L[j+1])
        for m in range(len(layer_after_merge)):
            t_after_merge = t_after_merge + pd_latency[str_][layer_after_merge[m]]
        # 合并
        if(t_max > t_after_merge):
            print("before merging: ")
            print("P: ", P)
            print("L: ", L)
            print("T: ", stage_time(L, P))
            P.insert(j+2, str_)
            P.pop(j)
            P.pop(j)
            num_stages[2] = num_stages[2] - 1
            num_stages[3] = num_stages[3] - 1
            print("after merging: ")
            print("P: ", P)
            L_copy = L[:]
            L_copy.insert(j+2, layer_after_merge)
            L_copy.pop(j)
            L_copy.pop(j)
            T_copy = stage_time(L_copy, P)
            print("L: ", L_copy)
            print("T: ", T_copy)
            print("after merging")
            print("P= ", P)
            L = work_flow(P, L_WL)
            T = stage_time(L, P)
            print("L= ", L)
            print("T= ", T)
            # j = j + 1
        else:
            j = j + 1
            # break
            print("Stop Further Merging!")
    # LOOP: Small Cluster
    k = num_stages[2]
    while (k < num_stages[3] - 1):
        # 计算原来两个stage的时间
        ts_1 = 0
        ts_2 = 0
        ts_max = 0
        for m in range(len(L[k])):
            ts_1 = ts_1 + pd_latency[P[k]][L[k][m]]
        for n in range(len(L[k+1])):
            ts_2 = ts_2 + pd_latency[P[k+1]][L[k+1][n]]
        ts_max = max(ts_1, ts_2)
        # 计算stage合并后的时间，确定是否需要合并
        ts_after_merge = 0
        changes = int(re.findall(r"\d+\.?\d*", P[k])[0]) + int(re.findall(r"\d+\.?\d*", P[k+1])[0])
        strs_ = str(changes) + "s"
        layers_after_merge = L[k][:]
        layers_after_merge.extend(L[k+1])
        for m in range(len(layers_after_merge)):
            ts_after_merge = ts_after_merge + pd_latency[strs_][layers_after_merge[m]]
        if (ts_max > ts_after_merge):
            print("before merging: ")
            print("P: ", P)
            print("L: ", L)
            print("T: ", stage_time(L, P))
            P.insert(k+2, strs_)
            P.pop(k)
            P.pop(k)
            num_stages[3] = num_stages[3] - 1
            print("after merging: ")
            print("P: ", P)
            Ls_copy = L[:]
            Ls_copy.insert(k + 2, layers_after_merge)
            Ls_copy.pop(k)
            Ls_copy.pop(k)
            print("L: ", Ls_copy)
            print("T: ", stage_time(Ls_copy, P))
            print("after merging")
            print("P= ", P)
            L = work_flow(P, L_WL)
            T = stage_time(L, P)
            print("L= ", L)
            print("T= ", T)
            # k = k + 1
        else:
            k = k + 1
            # break
            print("Stop Further Merging!")
    return L, P


# 优化后的pipt-it算法
# L_WL = []
# for j  in range(pd_latency.shape[0]):
#     L_WL.append(j)
# L, P = merge_stage(L_WL, 4, 4)
# T = stage_time(L, P)
# print(L)
# print(P)
# print(T)


# 暴力求解流水线算法
# start = time.perf_counter()
#
# L_WL = []
# for j  in range(pd_latency.shape[0]):
#     L_WL.append(j)
# best_pipeline = []
# best_combination = []
# best_stage_time = []
# min_maxtime = 10000
# min_E = 10000
# for j in range(len(pipeline)):
#     L = work_flow(pipeline[j], L_WL)
#     T = stage_time(L, pipeline[j])
#     print("P= ", pipeline[j])
#     print("L= ", L)
#     print("T= ", T)
#
#     sum_time = 0
#     for m in range(len(T)):
#         sum_time = sum_time + T[m]
#     avg_time = sum_time / len(T)
#     print("avg_time: ", avg_time)
#     E = 0
#     for n in range(len(T)):
#         E = E + (avg_time - T[n]) ** 2
#     E = E / len(T)
#     print("max_stage_time= ", max(T))
#     print("方差为：         ", E)
#     if (min_maxtime > max(T)):
#         min_maxtime = max(T)
#         min_E = E
#         best_pipeline = pipeline[j]
#         best_stage_time = T
#         best_combination = L
#     elif (min_maxtime == max(T) and min_E > E):
#         min_maxtime = max(T)
#         min_E = E
#         best_pipeline = pipeline[j]
#         best_stage_time = T
#         best_combination = L
# print("\n")
# print("best_pipeline: ", best_pipeline)
# print("best_combination: ", best_combination)
# print("best_stage_time: ", best_stage_time)
# print("max_time: ", min_maxtime)
# print("min_E: ", min_E)
#
# end = time.perf_counter()
# print("**********************\n", end - start)


# 论文中的算法和结果
# AlexNet
# L = [[0, 1], [2, 3, 4], [5], [6, 7]]
# P = ["gpu", "3b", "1b", "4s"]
# T = stage_time(L, P)
# print(T)
# [40.9123339, 40.583843200000004, 29.516859999999998, 7.208455000000001]
# GoogLeNet
# L = [[0, 1, 2, 3], [4, 5, 6], [7], [8], [9], [10, 11, 12]]
# P = ["gpu", "4b", "1s", "1s", "1s", "1s"]
# T = stage_time(L, P)
# print(T)
# [41.99017408, 51.1898153, 43.53859705, 43.53859705, 52.21434596, 56.33807396]
# ResNet
# L = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17]]
# P = ["gpu", "4b", "4s"]
# T = stage_time(L, P)
# print(T)
# [73.573553905, 92.24365940999999, 64.56268673]
# MobileNet
# L = [[0, 1, 2, 3], [4, 5], []]
# P = ["gpu", "2b", "2b", "1s", "1s", "1s", "1s"]
# T = stage_time(L, P)
# SqueezeNet
# L = [[0, 1, 2], [3], [4], [5], [6], [7, 8, 9]]
# P = ["gpu", "1b", "1b", "1b", "1b", "4s"]
# T = stage_time(L, P)
# print(T)
# [32.799881275000004, 56.03304332, 14.398945900000001, 19.15681542, 19.15681542, 29.634029650000002]
