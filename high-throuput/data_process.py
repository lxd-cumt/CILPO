import csv
# Iw,Id,Kw,Ow,N,M,K,time


csvFile = open("high-throu-traindata.csv", "r")
reader = csv.reader(csvFile) 
data = []
for item in reader:
    data.append(item)
csvFile.close()

f1 = open('ht-cpu-s1.csv','wb')
f2 = open('ht-cpu-s2.csv','wb')
f3 = open('ht-cpu-s3.csv','wb')
f4 = open('ht-cpu-s4.csv','wb')
f5 = open('ht-cpu-b1.csv','wb')
f6 = open('ht-cpu-b2.csv','wb')
f7 = open('ht-cpu-b3.csv','wb')
f8 = open('ht-cpu-b4.csv','wb')



m = len(data)
for i in range(m):
    if i%8==0:
        writer = csv.writer(f1)
    if i%8==1:
        writer = csv.writer(f2)
    if i%8==2:
        writer = csv.writer(f3)
    if i%8==3:
        writer = csv.writer(f4)
    if i%8==4:
        writer = csv.writer(f5)
    if i%8==5:
        writer = csv.writer(f6)
    if i%8==6:
        writer = csv.writer(f7)
    if i%8==7:
        writer = csv.writer(f8)

    writer.writerow(data[i])
