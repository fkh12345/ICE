
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
import torchvision.models as models

_HOST = '127.0.0.1'
_PORT = '8080'
def profile_model():
    model = models.vgg19()
    model.set_profile(True)
    model.set_input([0, 40])
    model.set_profile(False)
    model_input = model.get_input()
    del model
    return model_input

model_input = profile_model()


start1 = 0
end1 = 43
start2 = 40
end2 = 43
data1 = model_input[0]
data2 = model_input[1]

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=2400)
parser.add_argument('--slice', type=str, default='false')

args = parser.parse_args()

duration = np.zeros(args.bs)
start_time = np.zeros(args.bs)

ratio_sum = np.array([
    [0.5,0.67,0.75,0.8,0.67,0.57,0.5,0.67,0.75],
    [0.5,0.33,0.25,0.2,0.17,0.14,0,0.33,0.5]
])

def run(query_id, query_type, p_device):
    qtime = 0
    start_time_id = time()
    if(query_type < p_device):
        if(args.slice == 'false'):
            duration[query_id] = 0.1513
            return
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = qtime + np.random.uniform(0.093, 0.098)
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
    #print(query, ' ', float(response.text) + qtime)
    #print(response.text)
 
if __name__ == '__main__':
    
    plist = []
    query_list = []
    ratio = []
    ratio_level = ratio_sum.shape[1]
    level_num = int(args.bs)/ratio_level
    sleep_time = []   
    ratio_list = None     
    if(args.slice == 'true'):
        ratio_list = ratio_sum[0,:]
    else:
        ratio_list = ratio_sum[1,:]
    for i in range(0, args.bs):
        query_list.append(random())    
        level = int(i/level_num)
        ratio.append(ratio_list[level])
        
        

    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i], ratio[i]))
        plist.append(p)
    
    peak_load = 395
    thread_start_time = 0.88846

    for i in range(0, args.bs):
        wait_time = (1000 / peak_load - thread_start_time) / 1000
        sleep_time.append(wait_time)


    for num, item in enumerate(plist):
        #start = time()
        item.start()
        # ICE
        sleep(sleep_time[num])

    for item in plist:
        item.join()
    origin_duration = duration
    qos_level = 27
    p99_list = []
    profile_num = int(level_num / 3)
    for i in range(0, qos_level):
        tmp_duration = duration[i*profile_num:(i+1)*profile_num]
        sort_duration = np.sort(tmp_duration)
        len_list = sort_duration.shape[0]
        p99 = sort_duration[int(len_list * 0.99) - 1]
        p99_list.append(p99)
    print(p99_list)