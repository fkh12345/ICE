import os
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
PYTHON1='/root/miniconda3/envs/slice/bin/python'

def start_server(path ,cmd):
    os.chdir(path)
    os.system(cmd)

def kill_server():
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'')
    proc = []
    result = result.readlines()
    for line in result:
        if(line == ''):
            break
        proc.append(line)
    for item in proc:
        if('\n' in item):
            proctmp = item[0:-1]
        else:
            proctmp = item
        pid = int(proctmp)
        cmd = 'kill -kill {}'.format(pid)
        os.system(cmd)
    time.sleep(1)

def run_baseline(lag, load):
    result_latency = []
    bs = [64, 64, 32, 128, 64, 32, 128]
    for num, benchmark in enumerate(tqdm(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert'])):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} > /dev/null'.format(PYTHON1, bs[num])
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load {} --lag {}'.format(PYTHON1, load, lag)
        for i in range(0, 6):
            os.popen(cmd + ' > /dev/null')
            time.sleep(3)
        tmp = []
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline()
            result = float(result.split(' ')[0])
            tmp.append(result)
            time.sleep(3)
        tmp = np.array(tmp)
        tmp = np.average(tmp)
        result_latency.append(tmp)
        print(benchmark, tmp)
        kill_server()
        kill_server()
        if(benchmark == 'dfcnn'):
            kill_server()
        os.chdir('..')
    result_latency = np.array(result_latency)
    return result_latency
result = []
for load in ['high', 'medium', 'low']:        
    w_result = run_baseline('true', load)
    wo_result = run_baseline('false', load)
    result.append(w_result)
    result.append(wo_result)

result = np.array(result)
df = pd.DataFrame(result.T)
df.to_csv('data.csv')