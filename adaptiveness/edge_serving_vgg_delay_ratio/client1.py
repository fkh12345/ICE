
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
start1 = 0
end1 = 43
qtime = 0
tensor1 = torch.rand(1, 3, 224, 224)
start2 = 40
end2 = 43
tensor2 = torch.rand(1, 4096)
buffer1 = io.BytesIO()
torch.save(tensor1, buffer1)
buffer1.seek(0)
data1 = buffer1.read()
buffer2 = io.BytesIO()
torch.save(tensor2, buffer2)
buffer2.seek(0)
data2 = buffer2.read()

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--ratio', type=float)
args = parser.parse_args()

duration = np.zeros(args.bs)
start_time = np.zeros(args.bs)
def run(query_id, query_type, query_band, query_p):
    qtime = 0
    q = query_band
    start_time_id = time()
    if(query_type < query_p):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = 0.09 + qtime + 0.25/q
        query = 1

        #qtime = 0.15 + random() * 0.01
        #duration[query_id] = qtime
        #start_time[query_id] = start_time_id
        #return
        
    else:
        start = start1
        end = end1
        tensor = data1
        qtime = qtime + 1.53/q
        query = 0
    
    
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)  
    client = data_pb2_grpc.FormatDataStub(channel=conn)  
    #start1 = time() 
    response = client.DoFormat(data_pb2.actionrequest(text=tensor, start=start, end=end))
    
    #end1 = time()
    #print(query, ' ', float(response.text) + time)
    duration[query_id] = float(response.text) + qtime
    start_time[query_id] = start_time_id
    #print(query, ' ', float(response.text) + qtime)
    #print(response.text)
 
if __name__ == '__main__':
    
    plist = []
    query_list = []
    band = np.random.uniform(30, 100, args.bs)
    ratio = args.ratio
    sleep_time = []        
    
    for i in range(0, args.bs):
        query_list.append(random())    

    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i], band[i], ratio))
        plist.append(p)
    
    peak_load = 500
    thread_start_time = 0.88846

    for i in range(0, args.bs):
        wait_time = (1000 / peak_load - thread_start_time) / 1000
        sleep_time.append(wait_time)


    for num, item in enumerate(plist):
        #start = time()
        item.start()
        # ICE
        sleep(sleep_time[num])
        #sleep(0.00083)
        #sleep(0.0017)
        #sleep(0.0034)
        #sleep(0.0045)

        # Baseline
        #sleep(0.0025)
        #sleep(0.002)
        #end = time()
        #print(end - start)
    for item in plist:
        item.join()
    df = pd.DataFrame(duration)
    df.to_csv('stepping_load.csv')
    origin_duration = duration
    duration = np.sort(duration)
    p99 = duration[int(args.bs * 0.99) - 1]
    avg_time = np.average(duration)
    throughput = args.bs / (np.max(start_time) - np.min(start_time))

    list_shape = origin_duration.shape[0]
    point = int(list_shape / 100)
    '''
    for i in range(0, point):
        start = int(i * 100)
        end = int((i + 1) * 100)
        latency = origin_duration[start:end]
        latency = np.sort(latency)
        len_latency = latency.shape[0]
        print(latency[int(len_latency * 0.99) - 1])
        #print(np.average(latency))
    '''
    print(p99, avg_time, throughput) 