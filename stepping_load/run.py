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
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'|head -1')
    pid = int(result.readline())
    os.system('kill -kill ' + str(pid))
    os.system('kill -kill ' + str(pid+1))

def run_fluctuating():
    result_latency = []
    bs = [64, 64, 32, 128, 64, 32, 128]
    for num, benchmark in enumerate(tqdm(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert'])):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method {} --progress true > /dev/null'.format(PYTHON1, bs[num], 'ICE')
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 3000 --slice {}'.format(PYTHON1, 'true')
        for i in range(0, 5):
            os.system(cmd + ' > /dev/null')
            time.sleep(2)
        df = pd.read_csv('stepping_load.csv')
        latency_ice = np.array(df['0'])

        cmd = '{} client1.py --bs 3000 --slice {}'.format(PYTHON1, 'false')
        for i in range(0, 1):
            os.system(cmd + ' > /dev/null')
        df = pd.read_csv('stepping_load.csv')
        latency_db = np.array(df['0'])
        latency = np.array([latency_ice, latency_db])
        result_latency.append(latency)
        kill_server()
        os.chdir('..')
    result_latency = np.concatenate(result_latency, 0)
    result_latency = result_latency.T
    df = pd.DataFrame(result_latency)
    df.to_csv('data.csv')
        
run_fluctuating()