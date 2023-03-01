
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
from dnn_model.lapsrn import Net
def profile_model():
    model = Net()
    model.set_profile(True)
    model.set_input([0, 39])
    model.set_profile(False)
    model_input = model.get_input()
    del model
    return model_input

model_input = profile_model()
start1 = 0
end1 = 47
start2 = 39
end2 = 47

data1 = model_input[0]
data2 = model_input[1]

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--slice', type=str)
args = parser.parse_args()

duration = np.zeros(args.bs)
start_time = np.zeros(args.bs, dtype=np.float128)
def run(query_id, query_type, p_device):
    qtime = 0
    if(query_type < p_device):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = qtime + np.random.uniform(0.055, 0.056)
        query = 1
        
    else:
        start = start1
        end = end1
        tensor = data1
        qtime = qtime + np.random.uniform(0, 0.0002)
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
    p_device = 0.5
    if(args.slice == 'false'):
        p_device = 0
    sleep_time = []
    for i in range(0, args.bs):
        query_list.append(random())
    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i], p_device, ))
        plist.append(p)

    peak_load = 900
    thread_start_time = 0.88846
    for i in range(0, args.bs):
        level_step = args.bs / 10
        level = int(i / level_step) + 1
        load = peak_load / 10 * level
        wait_time = (1000 / load - thread_start_time) / 1000
        sleep_time.append(wait_time)

    for num, item in enumerate(plist):
        #start = time()
        item.start()
        # ICE
        sleep(sleep_time[num])
    for item in plist:
        item.join()
    result = []
    for i in range(0, 30):
        starttmp = i * 100
        endtmp = (i+1) * 100
        tmp = duration[starttmp:endtmp]
        pos = np.argmax(tmp)
        tmp = np.delete(tmp, pos)
        result.append(tmp)
    duration = np.concatenate(result, 0)
    df = pd.DataFrame(duration)
    df.to_csv('stepping_load.csv')
    duration = np.sort(duration)
    p99 = duration[int(args.bs * 0.99) - 1]
    print(p99) 