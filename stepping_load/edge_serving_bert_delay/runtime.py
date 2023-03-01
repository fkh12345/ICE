import numpy as np
import csv


file = open('profile/bert.csv','r')
time_layer = []
csv_reader = csv.reader(file)
for line in csv_reader:
    if('\ufeff' in line[0]):
        line[0] = line[0][1:-1] + line[0][-1]
    layer_time = float(line[0])
    time_layer.append(layer_time)


def change_waiting_queue(start, waiting_queue, query_start):
    # start records the start position of each query 
    # for example start = [0, 60, 0, 0, 60, 120, 120, 0, 0]
    # waiting_queue records the total waiting time of each query
    # query records each type of the queries
    # for example query_start = [0, 60, 120]
    query = [0, 0]
    for item in start:
        if(item == 0):
            query[0] = query[0] + 1
            query[1] = query[1] + 1
        else:
            query[1] = query[1] + 1 
    time1 = []
    start1 = 0
    time2 = 0
    for start2 in query_start:
        query_bs = query[start1]
        for layer in range(start1, start2):
            time2 = time2 + time_layer[layer]/128 * query_bs /1000
        start1 = start2
        time1.append(time2)
    #print(time1)
    for i, item in enumerate(start):
        if(item == 22):
            waiting_queue[i] = waiting_queue[i] - time1[1]
    return waiting_queue