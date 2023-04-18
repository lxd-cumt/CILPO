import csv
csvFile = open('delete.csv', "r")
reader = csv.reader(csvFile) 
data = []
_input = []
for item in reader:
    data.append(item)
csvFile.close()

id = 0
for item in data:
    if id >=5:
        _input.append(str(item)[2:50].lstrip().split(" "))
    id = id +1


# for i in range(len(_input)):
#     print (_input[i][0])

print "================"
# 6 states
input_data = []
id = 0
while id < 10:
    input_data.append(float(_input[id][0])/float(_input[id+1][0]))
    id = id + 2

input_data.append(float(_input[id][0])/float(_input[len(_input)-1][0])/1000)

print input_data