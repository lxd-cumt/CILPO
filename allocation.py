# -*- coding: utf-8 -*
import csv
import numpy as np
import pandas as pd
import re

###数据

file="pred"
#
layernum=8
layerset=[3,3,2]
pd_latency = pd.read_csv('schedule/'+file+'/'+'alex_latency.csv')


# layernum=57
# layerset=[19,19,19]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'gg_latency.csv')

# layernu=13
# layerset=[5, 4, 4]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'gg_latency1.csv')


# layernum=18
# layerset=[6,6,6]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'res_latency1.csv')


# layernum=54
# layerset=[18,18,18]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'res_latency.csv')


# layernum=28
# layerset=[10,9,9]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'mobi_latency.csv')


# layernum=15
# layerset=[5,5,5]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'mobi_latency1.csv')


# layernum=26
# layerset=[9,9,8]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'squ_latency.csv')

# layernum=10
# layerset=[4,3,3]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'squ_latency1.csv')


############################################################################ linear-pred
# file = "linear-pred"
# file = "linear-pred"


# layernum=8
# layerset=[3,3,2]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'alex.csv')

# layernum=12
# layerset=[4,4,4]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'gg.csv')

# layernum=18
# layerset=[6,6,6]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'res.csv')

# layernum=11
# layerset=[4,4,3]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'squ.csv')

# layernum=28
# layerset=[10,9,9]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'mobi.csv')
############################################################################

# file = "prepare+layer"

# layernum=8
# layerset=[3,3,2]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'alex_pred_latency.csv')

# layernum=12
# layerset=[4,4,4]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'gg_pred_latency.csv')

# layernum=18
# layerset=[6,6,6]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'res_pred_latency.csv')

# layernum=10
# layerset=[4,3,3]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'squ_pred_latency.csv')

# layernum=28
# layerset=[10,9,9]
# pd_latency = pd.read_csv('schedule/'+file+'/'+'mobi_pred_latency.csv')

#初始化
def init():
    list=[]
    list.append("gpu")
    list.append(layerset[0])
    gpu.append(list)

    if(layerset[1]>=4):
        for i in range(4):
            list=[]
            list.append("1b")
            if(i<3):
                list.append(int(layerset[1]/4))
            else:
                list.append(int(layerset[1]/4+layerset[1]%4))
            cpub.append(list)
    else:
        if (layerset[1]==3):
            list=[]
            list.append("2b")
            list.append(1)
            cpub.append(list)
            list=[]
            list.append("1b")
            list.append(1)
            cpub.append(list)
            list=[]
            list.append("1b")
            list.append(1)
            cpub.append(list)
        elif (layerset[1]==2):
            list=[]
            list.append("2b")
            list.append(1)
            cpub.append(list)
            list=[]
            list.append("2b")
            list.append(1)
            cpub.append(list)
        else:
            list=[]
            list.append("4b")
            list.append(1)
            cpub.append(list)

    if(layerset[2]>=4):
        for i in range(4):
            list=[]
            list.append("1s")
            if(i<3):
                list.append(int(layerset[2]/4))
            else:
                list.append(int(layerset[2]/4+layerset[2]%4))
            cpus.append(list)
    else:
        if (layerset[2]==3):
            list=[]
            list.append("1s")
            list.append(1)
            cpus.append(list)
            list=[]
            list.append("1s")
            list.append(1)
            cpus.append(list)
            list=[]
            list.append("2s")
            list.append(1)
            cpus.append(list)
        elif (layerset[2]==2):
            list=[]
            list.append("2s")
            list.append(1)
            cpus.append(list)
            list=[]
            list.append("2s")
            list.append(1)
            cpus.append(list)
        else:
            list=[]
            list.append("4s")
            list.append(1)
            cpus.append(list)


def cal_time():
    layer_=0
    #对于gpu
    temp=0
    list=[]
    for i in range(gpu[0][1]):
        temp+=pd_latency[gpu[0][0]][layer_]
        layer_+=1
    list.append(temp)
    time.append(list)
    #对于cpub
    list=[]
    for i in range(len(cpub)):
        temp=0
        for j in range(int(cpub[i][1])):
            temp+=pd_latency[cpub[i][0]][layer_]
            layer_+=1
        list.append(temp)
    time.append(list)
    #对于cpus
    list=[]
    for i in range(len(cpus)):
        temp=0
        for j in range(cpus[i][1]):
            temp+=pd_latency[cpus[i][0]][layer_]
            layer_+=1
        list.append(temp)
    time.append(list)

def alloc_b(cputime,start):
    if(len(cputime)==1):
        return
    if(max(cputime)-min(cputime))<10:
        return
    # cpub---time[1]
    recordstart=start

    count=0
    while count<=5:
        count+=1
        i=0
        start=recordstart
        flag=0
        while(i<len(cpub)-1):
            
            change=int(re.findall(r"\d+\.?\d*",cpub[i][0])[0])+int(re.findall(r"\d+\.?\d*",cpub[i+1][0])[0])
            str_=str(change)+"b"
            layer=cpub[i][1]+cpub[i+1][1]

            if i>0 and flag==0: 
                start+=cpub[i-1][1]

            #计算延迟
            latency_=max(cputime[i],cputime[i+1])
            cmp_laten=0
            for j in range(start,start+layer):
                cmp_laten+=pd_latency[str_][j]
            
            #合并 
            if(latency_>cmp_laten):
                #分配改完了
                flag=1
                cpub[i][0]=str_
                cpub[i][1]=layer
                cpub.pop(i+1)   
                #latency改
                cputime[i]=cmp_laten
                cputime.pop(i+1)
            else:
                flag=0
                i+=1
        #end while
    #end count
    time[1]=cputime

def alloc_s(cputime,start):
    if(len(cputime)==1):
        return
    if(max(cputime)-min(cputime))<10:
        return
    # cpub---time[1]
    recordstart=start

    count=0
    while count<=5:
        count+=1
        i=0
        start=recordstart
        flag=0
        while(i<len(cpus)-1):            
            change=int(re.findall(r"\d+\.?\d*",cpus[i][0])[0])+int(re.findall(r"\d+\.?\d*",cpus[i+1][0])[0])
            str_=str(change)+"s"
            layer=cpus[i][1]+cpus[i+1][1]
            if i>0 and flag==0:
                start+=cpus[i-1][1]

            #计算延迟
            latency_=max(cputime[i],cputime[i+1])
            cmp_laten=0
            for j in range(start,start+layer):
                cmp_laten+=pd_latency[str_][j]
            
            #合并
            if(latency_>cmp_laten ):
                #分配改完了
                flag=1
                cpus[i][0]=str_
                cpus[i][1]=layer
                cpus.pop(i+1)   
                #latency改
                cputime[i]=cmp_laten
                cputime.pop(i+1)
            else:
                flag=0
                i+=1
        #end while
    #end count
    time[2]=cputime


gpu=[]
cpub=[]
cpus=[]
time=[]

init()
cal_time()

print (gpu)
print (cpub)
print (cpus)
print (time)

print ("给大核重新分配")
alloc_b(time[1],layerset[0])
print ("分配完了")
print ("给xiao核重新分配")
alloc_s(time[2],layerset[0]+layerset[1])
print (gpu)
print (cpub)
print (cpus)
print (time)

# print "max="+str(max(time))


# #如果gpu time 大出来10
count=0
while(count<=20):
    print ("count")
    count+=1
    flag=0
    if(flag==0):
        if((max(time[0])-max(time[1]))>7 and layerset[0]>1):
            flag=1
            layerset[0]-=1
            layerset[1]+=1
            if((max(time[1])-max(time[2]))>10 and layerset[1]>1):
                layerset[1]-=1
                layerset[2]+=1
    if(flag==0):
        if((max(time[2])-max(time[1]))>10 and layerset[2]>1):
            flag=1
            layerset[1]+=1
            layerset[2]-=1
            if((max(time[1])-max(time[0]))>10 and layerset[1]>1):
                layerset[1]-=1
                layerset[0]+=1
    if(flag==0):
        if((max(time[1])-max(time[2]))>10 and layerset[1]>1):
            flag=1
            layerset[2]+=1
            layerset[1]-=1
            if((max(time[2])-max(time[0]))>10 and layerset[2]>1):
                layerset[2]-=1
                layerset[0]+=1

    if(flag==1):

        gpu=[]
        cpub=[]
        cpus=[]
        time=[]
        init()
        print (gpu)
        print (cpub)
        print (cpus)

        cal_time()
        
        print ("给大核重新分配")
        alloc_b(time[1],layerset[0])
        
        print ("给小核重新分配")
        alloc_s(time[2],layerset[0]+layerset[1])
        
        print (gpu)
        print (cpub)
        print (cpus)
        print (layerset)
        print (time)
        print ("max="+str(max(time)))


# # P = ["gpu", "1b", "1b", "1b", "1b", "1s", "1s", "1s", "1s"]
# P = ["1b", "1b", "1b", "1b", "1s", "1s", "1s", "1s"]
# L_WL = []
# for j  in range(pd_latency.shape[0]):
#     L_WL.append(j)
# L_res = work_flow(P, L_WL)
# T = []
# for j in range (len(L_res)):
#     time=0
#     for k in range (len(L_res[j])):
#         time = time + pd_latency[P[j]][L_res[j][k]]
#     T.append(time)
# print(P)
# print(L_res)
# print(T)