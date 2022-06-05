
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
start1 = 0
end1 = 173
qtime = 0
tensor1 = torch.rand(1, 3, 224, 224)
start2 = 109
end2 = 173
tensor2 = torch.rand(1, 1024, 14, 14)
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

cloud_time = np.zeros((args.bs, 1))
edge_time = np.zeros((args.bs, 1))
tran_time = np.zeros((args.bs, 1))
wait_time_1 = np.zeros((args.bs, 1))
wait_time_2 = np.zeros((args.bs, 1))


def guarantee_ratio(duration):
    len_duration = duration.shape[0]
    l2 = np.where(duration > 0.2 * 1.02)[0].shape[0]
    l10 = np.where(duration > 0.2 * 1.1)[0].shape[0]
    l20 = np.where(duration > 0.2 * 1.2)[0].shape[0]
    l30 = np.where(duration > 0.2 * 1.3)[0].shape[0]
    l2 = (l2 - l10)/len_duration
    l10 = (l10 - l20)/len_duration
    l20 = (l20 - l30)/len_duration
    l30 = l30/len_duration
    return l2, l10, l20, l30

def run(query_id, query_type, query_band):
    qtime = 0
    q = query_band
    if(query_type < 0.25):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        qtime = 0.03 + qtime
        edge_time[query_id] = 0.043
        tran_time[query_id] = 0.51/q
        query = 1
        
    else:
        start = start1
        end = end1
        tensor = data1
        edge_time[query_id] = 0
        tran_time[query_id] = 1.53/q
        query = 0
    
    
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)  
    client = data_pb2_grpc.FormatDataStub(channel=conn)  
    #start1 = time() 
    
    response = client.DoFormat(data_pb2.actionrequest(text=tensor, start=start, end=end))
    start_time_id = time()
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
    band = np.random.uniform(100, 300, args.bs)

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
            sleep(0.0003)
        elif(args.load == 'medium'):
            sleep(0.0015)
        else:
            sleep(0.002)

        # Baseline
        #sleep(0.005)
        #sleep(0.003)
        #sleep(0.0025)
        #sleep(0.002)
        #end = time()
        #print(end - start)
    for item in plist:
        item.join()
    
    duration = np.sort(duration)
    l2, l10, l20, l30 = guarantee_ratio(duration)
    print(l2, l10, l20, l30)
    p99 = duration[int(args.bs * 0.99) - 1]
    avg_time = np.average(duration)
    throughput = args.bs / (np.max(start_time) - np.min(start_time))

    scale_result = [l2, l10, l20, l30 ,throughput]
    file = open('scale.csv', 'a', newline='')
    csv_file = csv.writer(file)
    csv_file.writerow(scale_result)
    file.close()

    breakdown = np.concatenate([cloud_time, edge_time, tran_time, wait_time_1, wait_time_2], axis=1)
    pos = np.where(breakdown[:,0] == -1.0)
    breakdown = np.delete(breakdown, pos, axis=0)
    breakdown_avg = np.average(breakdown, axis=0)
    breakdown = np.concatenate([breakdown, [breakdown_avg]], axis=0)
    df = pd.DataFrame(breakdown)
    df.columns=['cloud_time','edge_time','tran_time', 'wait_time_1', 'wait_time_2']
    df.to_csv('breakdown.csv')


    print(p99, avg_time, throughput) 
    #print(start_time)