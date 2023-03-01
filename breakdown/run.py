import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp

PYTHON1='/root/miniconda3/envs/slice/bin/python'
PYTHON2='/root/miniconda3/envs/no-batch/bin/python'

def start_server(path ,cmd):
    os.chdir(path)
    os.system(cmd)

def kill_server():
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'|head -1')
    pid = int(result.readline())
    os.system('kill -kill ' + str(pid))
    os.system('kill -kill ' + str(pid+1))

def run_baseline1(method):
    bs = [64, 64, 32, 128, 64, 32, 128]
    result_latency = []
    for num, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method {} --progress true > /dev/null'.format(PYTHON1, bs[num], method)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300'.format(PYTHON1)
        latency = []
        for i in range(0, 3):
            os.system(cmd + ' > /dev/null')
            time.sleep(1.5)
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline()
            result = result.split(' ')
            breakdown = [float(result[0]), float(result[1]), float(result[2]), float(result[3])]
            latency.append(breakdown)
            time.sleep(1.5)
        latency = np.array(latency)
        print(benchmark, np.average(latency, 0))
        result_latency.append(np.average(latency, 0))
        kill_server()
        #p.close()
        os.chdir('..')
    return result_latency

def run_baseline2():
    result_latency = []
    for num, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = 'edge_serving_{}'.format(benchmark)
        server_cmd = '{} server.py > /dev/null'.format(PYTHON2)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 180'.format(PYTHON2)
        latency = []
        for i in range(0, 3):
            os.system(cmd + ' > /dev/null')
            time.sleep(1.5)
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline()[1:-1]
            result = result.split(' ')
            breakdown = [float(result[0]), float(result[1]), float(result[2]), float(result[3])]
            latency.append(breakdown)
            time.sleep(1.5)
        latency = np.array(latency)
        print(benchmark, np.average(latency, 0))
        result_latency.append(np.average(latency, 0))
        kill_server()
        #p.close()
        os.chdir('..')
    return result_latency

os.chdir('slicing')
breakdown_slicing = run_baseline1('ICE')
os.chdir('..')
os.chdir('without_slicing')
breakdown_db = run_baseline1('DB')
os.chdir('..')
os.chdir('without_batching')
breakdown_neu = run_baseline2()
os.chdir('..')

breakdown = np.concatenate([breakdown_slicing, breakdown_db, breakdown_neu], 1)
df = pd.DataFrame(breakdown)
df.to_csv('data.csv')