import os
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
PYTHON1='/root/miniconda3/envs/slice/bin/python'
PYTHON2='/root/miniconda3/envs/no-batch/bin/python'

def start_server(path ,cmd):
    os.chdir(path)
    os.system(cmd)

def kill_server():
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'|head -2')
    proc_list = []
    proc_list.append(int(result.readline()[0:-1]))
    proc_list.append(int(result.readline()[0:-1]))
    for item in proc_list:
        os.system('kill -kill ' + str(item) + ' > /dev/null')
        os.system('kill -kill ' + str(item + 1) + ' > /dev/null')

    #os.system('kill -kill ' + str(pid+1))

def run_baseline1(method, slice):
    os.chdir('batching')
    bs = [64, 64, 32, 128, 64, 32, 128]
    result_latency = []
    for num, benchmark in enumerate(tqdm(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert'])):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method {} --progress true --worker 0 > /dev/null'.format(PYTHON1, bs[num], method)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        server_cmd = '{} server.py --bs {} --method {} --progress true --worker 1 > /dev/null'.format(PYTHON1, bs[num], method)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = 'bash test.sh {}'.format(slice)
        latency = []
        for i in range(0, 4):
            os.system(cmd + ' > /dev/null')
            time.sleep(0.5)
        result_sum = []
        for i in range(0, 4):
            result = os.popen(cmd)
            tmp = result.readline().split(' ')
            for j, item in enumerate(tmp):
                tmp[j] = float(tmp[j])
            result_sum.append(tmp)
            time.sleep(0.5)
        latency = np.array(result_sum)
        latency = np.average(latency, 0)
        result_latency.append(latency)
        kill_server()
        kill_server()
        #p.close()
        os.chdir('..')
    result_latency = np.array(result_latency)
    result_latency = np.average(result_latency, 0)
    os.chdir('..')
    return result_latency

def run_baseline2():
    os.chdir('without_batching')
    result_latency = []
    for benchmark in tqdm(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = '../without_batching/edge_serving_{}'.format(benchmark)
        server_cmd = '{} server.py --worker 0 > /dev/null'.format(PYTHON2)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        server_cmd = '{} server.py --worker 1 > /dev/null'.format(PYTHON2)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = 'bash test.sh'
        latency = []
        for i in range(0, 4):
            os.system(cmd + ' > /dev/null')
            time.sleep(0.5)
        result_sum = []
        for i in range(0, 4):
            result = os.popen(cmd)
            tmp = result.readline().split(' ')
            for j, item in enumerate(tmp):
                tmp[j] = float(tmp[j])
            result_sum.append(tmp)
            time.sleep(0.5)
        latency = np.array(result_sum)
        latency = np.average(latency, 0)
        result_latency.append(latency)
        kill_server()
        kill_server()
        os.chdir('..')
    result_latency = np.array(result_latency)
    result_latency = np.average(result_latency, 0)
    os.chdir('..')
    return result_latency

result_ice = run_baseline1('ICE', 'true')
result_db = run_baseline1('DB', 'false')
result_neu = run_baseline2()

data = np.array([result_ice, result_db, result_neu])
data = data.T

df = pd.DataFrame(data)
df.to_csv('data.csv')

