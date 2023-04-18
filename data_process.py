import csv
# Iw,Id,Kw,Ow,N,M,K,time


csvFile = open('dataset-train/col2im-set.csv', "r")
reader = csv.reader(csvFile) 
data = []
for item in reader:
    data.append(item)
csvFile.close()

f1 = open('cpu-s1-col2im.csv','wb')
f2 = open('cpu-s2-col2im.csv','wb')
f3 = open('cpu-s3-col2im.csv','wb')
f4 = open('cpu-s4-col2im.csv','wb')
f5 = open('cpu-b1-col2im.csv','wb')
f6 = open('cpu-b2-col2im.csv','wb')
f7 = open('cpu-b3-col2im.csv','wb')
f8 = open('cpu-b4-col2im.csv','wb')



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
