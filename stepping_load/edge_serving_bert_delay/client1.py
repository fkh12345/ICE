
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
    #del result
    del model
    #del data
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
parser.add_argument('--bs', type=int, default=300)
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
    sleep_time = []
    p_device = 0.5
    if(args.slice == 'false'):
        p_device = 0
    peak_load = 655
    thread_start_time = 0.88846
    level_start = 5
    level_sum = 10
    for i in range(0, args.bs):
        level_step = args.bs / (level_sum - level_start)
        level = int(i / level_step) + level_start - 0.25
        load = peak_load / level_sum * level
        wait_time = (1000 / load - thread_start_time) / 1000
        sleep_time.append(wait_time)

    for i in range(0, args.bs):
        query_list.append(random())
    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i], p_device, ))
        plist.append(p)
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
