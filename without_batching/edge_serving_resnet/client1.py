
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
import csv
_HOST = '127.0.0.1'
_PORT = '8080'
import torchvision.models as models
def profile_model():
    model = models.resnet50()
    model.set_profile(True)
    model.set_input([0, 112])
    model.set_profile(False)
    model_input = model.get_input()
    del model
    return model_input

model_input = profile_model()

start1 = 0
end1 = 173
start2 = 109
end2 = 173
data1 = model_input[0]
data2 = model_input[1]
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--load', type=str, default='high')
args = parser.parse_args()


duration = np.zeros(args.bs)
start_time = np.zeros(args.bs, dtype=np.float128)

def run(query_id, query_type):
    qtime = 0
    if(query_type < 0.5):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = qtime + np.random.uniform(0.048, 0.060)
        query = 1
        
    else:
        start = start1
        end = end1
        tensor = data1
        qtime = qtime + np.random.uniform(0.015, 0.051)
        query = 0
    
    
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)  
    client = data_pb2_grpc.FormatDataStub(channel=conn)  
    #start1 = time() 
    response = client.DoFormat(data_pb2.actionrequest(text=tensor, start=start, end=end))
    
    #end1 = time()
    #print(query, ' ', float(response.text) + time)
    duration[query_id] = float(response.text) + qtime
    start_time_id = time()
    start_time[query_id] = start_time_id
    
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
        if(args.load == 'high'):
            sleep(0.0006)
        elif(args.load == 'medium'):
            sleep(0.0015)
        elif(args.load == 'low'):
            sleep(0.002)
        else:
            sleep(0.0016)
    for item in plist:
        item.join()
    
    duration = np.sort(duration)
    p99 = duration[int(args.bs * 0.99) - 1]
    avg_time = np.average(duration)
    throughput = args.bs / (np.max(start_time) - np.min(start_time))

    print(p99, avg_time, throughput) 
