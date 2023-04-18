# -*- coding: utf-8 -*-
import csv
import numpy as np
import random
import os
import time


f = open('espoid_1.csv','wb')
csv_writer = csv.writer(f)


inputs = []
writes = []
def process_input():
    global writes
    csvFile = open('delete.csv', "r")
    reader = csv.reader(csvFile) 
    data = []
    _input = []
    for item in reader:
        data.append(item)

    csvFile.close()

    # f = open("delete.csv", "w")
    # f.truncate()
    # f.close()

    id = 0
    for item in data:
        if id >=5:
            _input.append(str(item)[2:50].lstrip().split(" "))
        id = id +1

   
    # imput_nums = 6
    id = 0
    while id < 10:
        inputs.append(float(_input[id][0])/float(_input[id+1][0]))
        id = id + 2

    inputs.append(float(_input[id][0])/float(_input[id+2][0])/1000)
    print("inputs= ",inputs)
    writes.append(inputs)
    writes.append(float(_input[id+2][0]))
    

def action(shid, a):
    if a==0:
        res = (shid + 1) % 21

    if a==1:
        res = (shid - 1 + 21) % 21

    if a==2:
        res = (shid + 7) % 21

    if a==3:   
        res = (shid - 7 + 21) % 21

    if a==4:
        res = (shid + 14) % 21

    if a==5:
        res = (shid - 14 + 21) % 21
    
    return res
    

def sigmoid(x):
    # our activation function: f(x) = 1 / (1 * e^(-x))
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


#设置 2个隐层 第一层12个神经元 第二层4-6个神经元
ww1 = np.array([[1.0077947, -0.8674698, -1.6896673, 0.51038975, -0.2786524, -0.6172485, -1.1154783,
                    1.6372956, -1.2183477, 0.65986705, -1.0881641, -0.6260325],
                [-0.19941649, -0.64269257, -0.49515888, 0.09501503, 0.33763403, 0.95985407, 1.5934367,
                    -0.8535863, 1.2862705, 0.9269908, -0.3461618,  0.10150014],
                [1.3806945, 1.1642591, -0.26192477, -0.960971, 0.13559048, 1.1402102, 0.44013, 0.03423633,
                    1.3417107, -1.4666897, -0.33415636, -0.6764755],
                [1.5018231, -1.2813721, -0.13244262, 0.4771869, 0.05031537, -1.5981716, 0.20770457,
                    0.3673019,-0.5660462, -0.9611329, 0.56620693, -1.6140602],
                [-0.04557747, 0.03933186, 0.85725987, -1.7968763, -1.1048409, 0.01140761, -0.13175301, 
                    -0.9273326, 1.4259161, 0.07385238, 0.835029,-0.24543218],
                [1.5690517, -0.33383316, 0.7228393, 1.4647387, -0.33378315, 0.2617152, -0.9381871,
                    0.43009415, -0.63503337, -1.2160896, -0.09295993,  0.7061125]])

ww2 = np.array(
    [[1.1546336, 0.27586943, 1.1079186, 0.55474895, 0.72035396, 0.43164253], 
        [0.24701223, 0.93802154, 0.6546343, -0.34011808, 0.72035396, 0.43164253],
        [0.92324615, 0.24938142, -0.9950161, 1.0493605, 0.72035396, 0.43164253],
        [-0.66868544, 0.11709079, 0.40426564, -0.36460695, 0.72035396, 0.43164253], 
        [-1.5412269, 0.25356004, 1.4325153, -0.7276282, 0.72035396, 0.43164253],
        [0.46994478, -0.23374167, 1.2352585, -0.5692175, 0.72035396, 0.43164253],
        [-1.5094587, 0.66745085, -0.7243643, -0.21625689, 0.72035396, 0.43164253], 
        [1.0044347, 0.69552636, -1.7080321, 0.6040948, 0.72035396, 0.43164253],
        [0.8460662, -0.84302175, 1.5234374, -0.7839487, 0.72035396, 0.43164253],
        [0.41058198, 0.26875126, 0.31888998, -0.26302937, 0.72035396, 0.43164253], 
        [-0.18953407, 1.125341, 0.95590675, -0.4819808, 0.72035396, 0.43164253],
        [-0.18220073, 0.29417682, 0.24189407, 1.0429344, 0.72035396, 0.43164253]])

bb1 = np.array([1.1217784, 1.2005006, - 0.33468056, 0.855867, 0.9069824, 1.2543672, - 0.11337583, - 0.8478395,
                0.6908875,  0.05043283, 0.8528839, - 1.9828062])

bb2 = np.array([0.1917277, 0.6224212, 0.5894047, 0.4795857, 0.05828818, 0.7951844])


def main():
    global writes
    
    global inputs
    # shid = 4
    # command = "perf stat -e instructions,cycles,cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-store-misses,L1-dcache-stores,L1-icache-load-misses,L1-icache-loads,cpu-clock -o delete.csv ./" +str(shid) +".sh "
    # os.system(command) 
    
    for step in range(8000):
        print ("step:" , step)
        if step > 0:
            process_input()
        
        else:
            shid = 18
            inputs = [0.5907227394068322, 0.02556810241497497, 0.025516994359938858, 0.02507397181851087, 0.020541758151494397, 4.2301453199889405]

        p = random.random()

        if 0.7 >= p >=0.2:
            index = random.randint(0,5)
        else:
            hide1 = np.dot(inputs, ww1) + bb1
            hide1 = sigmoid(hide1)

            hide2 = np.dot(hide1, ww2) + bb2
            hide2 = sigmoid(hide2)

            res = softmax(hide2)
            index = res.tolist().index(max(res))
        
        inputs = []
        writes.append(index)
        csv_writer.writerow(writes)
        writes = []
        res = action(shid, index)
        shid = res
        print("index=",index,"execID=",res)
        # exec sh
        command = "perf stat -e instructions,cycles,cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-store-misses,L1-dcache-stores,L1-icache-load-misses,L1-icache-loads,cpu-clock -o delete.csv ./" +str(res) +".sh "
        os.system(command)


if __name__ == '__main__':
  main()
