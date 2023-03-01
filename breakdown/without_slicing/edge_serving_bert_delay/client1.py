
import grpc
import inference_pb2 as data_pb2
import inference_pb2_grpc as data_pb2_grpc
from multiprocessing import Process
from threading import Thread
import torch
import io
import argparse
import numpy as np
from time import time, time_ns
from time import sleep
#string = base64.b64encode(buffer)
from random import random
import pandas as pd
_HOST = '127.0.0.1'
_PORT = '8080'
from dnn_model.ner import My_NER
def profile_model():
    model = My_NER()
    model.set_profile(True)
    model.set_input([0, 22])
    model.set_profile(False)
    model_input = model.get_input()
    del model
    sleep(0.005)
    return model_input

start1 = 0
end1 = 131
start2 = 22
end2 = 131

model_input = profile_model()
data1 = model_input[0]
data2 = model_input[1]

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
args = parser.parse_args()

duration = np.zeros(args.bs)
start_time = np.zeros(args.bs, dtype=np.float128)

cloud_time = np.zeros((args.bs, 1))
edge_time = np.zeros((args.bs, 1))
tran_time = np.zeros((args.bs, 1))
wait_time_1 = np.zeros((args.bs, 1))
wait_time_2 = np.zeros((args.bs, 1))



def run(query_id, query_type):
    qtime = 0
    if(query_type < 0):
        return
        
    else:
        start = start1
        end = end1
        tensor = data1
        qtime = np.random.uniform(0.0019, 0.0038)
        edge_time[query_id] = 0
        tran_time[query_id] = qtime - edge_time[query_id]
        query = 0
    
    
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)  
    client = data_pb2_grpc.FormatDataStub(channel=conn)  
    #start1 = time() 
    start_time_id = time()
    response = client.DoFormat(data_pb2.actionrequest(text=tensor, start=start, end=end))
    #end1 = time()
    #print(query, ' ', float(response.text) + time)
    duration[query_id] = float(response.text) + qtime
    start_time[query_id] = start_time_id
    cloud_time[query_id] = float(response.queue)
    wait_time_1[query_id] = duration[query_id] - cloud_time[query_id] - edge_time[query_id] - tran_time[query_id]
    wait_time_2[query_id] = float(response.text) - cloud_time[query_id]
    #print(query, ' ', float(response.text) + qtime)
    #print(response.text)
 
if __name__ == '__main__':
    
    plist = []
    query_list = []
    for i in range(0, args.bs):
        query_list.append(random())
    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i]))
        plist.append(p)
    for item in plist:
        #start = time()
        item.start()
        # ICE
        sleep(0.00065)

    for item in plist:
        item.join()
    
    duration = np.sort(duration)
    avg_time = np.average(duration)
    throughput = args.bs / (np.max(start_time) - np.min(start_time))
    breakdown = np.concatenate([cloud_time, edge_time, tran_time, wait_time_1, wait_time_2], axis=1)
    breakdown_avg = np.average(breakdown, axis=0)

    print(breakdown_avg[0],breakdown_avg[1],breakdown_avg[2],breakdown_avg[3])
