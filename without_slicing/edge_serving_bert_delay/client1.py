
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
_HOST = '127.0.0.1'
_PORT = '8080'
start1 = 0
end1 = 131
qtime = 0
tensor1 = torch.rand(1, 32, 768)
start2 = 23
end2 = 131
tensor2 = torch.rand(1, 32, 768)
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
parser.add_argument('--load', type=str, default='high')
args = parser.parse_args()
duration = np.zeros(args.bs)
start_time = np.zeros(args.bs, dtype=np.float128)
def run(query_id, query_type, query_band):
    qtime = 0
    q = query_band
    if(query_type < 0.5):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = 0.008 + qtime
        query = 1
        
    else:
        start = start1
        end = end1
        tensor = data1
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
    #print(query, ' ', float(response.text) + qtime)
    #print(response.text)
 
if __name__ == '__main__':
    
    plist = []
    query_list = []
    band = np.random.uniform(50, 150, args.bs)

    for i in range(0, args.bs):
        query_list.append(random())
    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i], band[i]))
        plist.append(p)
    for item in plist:
        #start = time()
        item.start()
        # ICE
        if(args.load == 'high'):
            sleep(0.00065)
        elif(args.load == 'medium'):
            sleep(0.001)
        else:
            sleep(0.002)
        #sleep(0.004)
        #sleep(0.0045)

        # Baseline
        #sleep(0.0008)
        #sleep(0.003)
        #sleep(0.0025)
        #sleep(0.002)
        #end = time()
        #print(end - start)
    for item in plist:
        item.join()
    
    duration = np.sort(duration)
    p99 = duration[int(args.bs * 0.99) - 1]
    avg_time = np.average(duration)
    throughput = args.bs / (np.max(start_time) - np.min(start_time))
    print(p99, avg_time, throughput) 
    #print(start_time)
