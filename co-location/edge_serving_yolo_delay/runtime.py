import numpy as np
import csv


file = open('profile/yolo.csv','r')
time_layer = []
csv_reader = csv.reader(file)
for line in csv_reader:
    layer_time = float(line[0])
    time_layer.append(layer_time)


def change_waiting_queue(start, waiting_queue, query_start):
    # start records the start position of each query 
    # for example start = [0, 60, 0, 0, 60, 120, 120, 0, 0]
    # waiting_queue records the total waiting time of each query
    # query records each type of the queries
    # for example query_start = [0, 60, 120]
    start_np = np.array(start)
    unique_start =np.unique(np.array(start_np))
    query = np.array([0 for _ in query_start])
    #query = [0, 0]
    for num, item in enumerate(query_start):
        bs = np.where(start_np == query_start[num])[0].shape[0]
        query[num:-1] = query[num:-1] + bs
        query[-1] = query[-1] + bs
        

    #for item in start:
    #    if(item == 0):
    #        query[0] = query[0] + 1
    #        query[1] = query[1] + 1
    #    else:
    #        query[1] = query[1] + 1 
    time1 = []
    start1 = 0
    time2 = 0
    if(query_start[0] == 0):
        query_start.pop(0)
    for num, start2 in enumerate(query_start):
        query_bs = query[num]
        for layer in range(start1, start2):
            time2 = time2 + time_layer[layer]/8 * query_bs /1000
        start1 = start2
        time1.append(time2)

    for i, item in enumerate(start):
        if(item == 85):
            waiting_queue[i] = waiting_queue[i] - time1[0]
        elif(item == 96):
            waiting_queue[i] = waiting_queue[i] - time1[1]
        
    return waiting_queue


