import os
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
PYTHON1='/root/miniconda3/envs/slice/bin/python'
PYTHON2='/root/miniconda3/envs/no-batch/bin/python'

def start_server(path ,cmd):
    os.chdir(path)
    os.system(cmd)

def kill_server():
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'')
    proc = []
    proc.append(result.readline())
    proc.append(result.readline())
    for item in proc:
        if('\n' in item):
            proctmp = item[0:-1]
        else:
            proctmp = item
        pid = int(proctmp)
        cmd = 'kill -kill {}'.format(pid)
        os.system(cmd)

def run_baseline1(method, slice, load):
    bs = [64, 64, 32, 128, 64, 32, 128]
    result_latency = []
    for num, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method {} --progress true > /dev/null'.format(PYTHON1, bs[num], method)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load {} --slice {}'.format(PYTHON1, load, slice)
        latency = []
        for i in range(0, 4):
            os.system(cmd + ' > /dev/null')
            time.sleep(1.5)
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline().split(' ')
            latency.append(float(result[0]))
            time.sleep(1.5)
        latency = np.array(latency)
        print(benchmark, np.average(latency))
        result_latency.append(np.average(latency))
        kill_server()
        kill_server()
        #p.close()
        os.chdir('..')
    return result_latency

def run_baseline2(load):
    result_latency = []
    for benchmark in ['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']:
        path = '../without_batching/edge_serving_{}'.format(benchmark)
        server_cmd = '{} server.py > /dev/null'.format(PYTHON2)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load {}'.format(PYTHON2, load)
        latency = []
        for i in range(0, 2):
            result = os.popen(cmd)
            result = result.readline().split(' ')
            latency.append(float(result[0]))
            time.sleep(1.5)
        latency = np.array(latency)
        result_latency.append(np.average(latency))
        print(benchmark, np.average(latency))
        kill_server()
        kill_server()
        #p.close()
        os.chdir('../../without_slicing')
    return result_latency

def run_baseline3():
    result_load1 = []
    result_load2 = []
    result_load3 = []
    for benchmark in ['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']:
        path = '../without_batching/edge_serving_{}'.format(benchmark)
        server_cmd = '{} server.py > /dev/null'.format(PYTHON2)
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load {}'.format(PYTHON2, 'peak')
        tmp = []
        for i in range(0, 2):
            result = os.popen(cmd)
            result = result.readline().split(' ')
            tmp.append(float(result[2]))
            time.sleep(1.5)
        tmp = np.array(tmp)
        result_load2.append(np.average(tmp))
        print(benchmark, np.average(tmp))
        kill_server()
        kill_server()
        #p.close()
        os.chdir('../../without_slicing')
    
    bs = [64, 64, 32, 128, 64, 32, 128]
    for num, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method DB --progress true > /dev/null'.format(PYTHON1, bs[num])
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load peak --slice false'.format(PYTHON1)
        tmp = []
        for i in range(0, 4):
            os.system(cmd + ' > /dev/null')
            time.sleep(1.5)
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline().split(' ')
            tmp.append(float(result[2]))
            time.sleep(1.5)
        tmp = np.array(tmp)
        print(benchmark, np.average(tmp))
        result_load1.append(np.average(tmp))
        kill_server()
        kill_server()
        #p.close()
        os.chdir('..')

    for num, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
        path = 'edge_serving_{}_delay'.format(benchmark)
        server_cmd = '{} server.py --bs {} --method ICE --progress true > /dev/null'.format(PYTHON1, bs[num])
        p = mp.Process(target=start_server, args=(path, server_cmd))
        p.start()
        time.sleep(20)
        os.chdir(path)
        cmd = '{} client1.py --bs 300 --load high --slice true'.format(PYTHON1)
        tmp = []
        for i in range(0, 3):
            os.system(cmd + ' > /dev/null')
            time.sleep(1.5)
        for i in range(0, 3):
            result = os.popen(cmd)
            result = result.readline().split(' ')
            tmp.append(float(result[2]))
            time.sleep(1.5)
        tmp = np.array(tmp)
        print(benchmark, np.average(tmp))
        result_load3.append(np.average(tmp))
        kill_server()
        kill_server()
        #p.close()
        os.chdir('..')
    
    return result_load1, result_load2, result_load3


l_high = run_baseline1('ICE', 'true', 'high')
l_medium = run_baseline1('ICE', 'true', 'medium')
l_low = run_baseline1('ICE', 'true', 'low')
l_ice = np.array([l_high, l_medium, l_low])


l_high = run_baseline1('DB', 'false', 'high')
l_medium = run_baseline1('DB', 'false', 'medium')
l_low = run_baseline1('DB', 'false', 'low')
l_db = np.array([l_high, l_medium, l_low])

l_high = run_baseline2('high')
l_medium = run_baseline2('medium')
l_low = run_baseline2('low')
l_neu = np.array([l_high, l_medium, l_low])

l_high = run_baseline1('DB', 'true', 'high')
l_medium = run_baseline1('DB', 'true', 'medium')
l_low = run_baseline1('DB', 'true', 'low')
l_np = np.array([l_high, l_medium, l_low])





load_db, load_neu, load_ice = run_baseline3()
load = np.array([load_db, load_neu, load_ice])

data = np.concatenate([l_ice, l_np, l_db, l_neu, load], 0)
data = data.T

df = pd.DataFrame(data)
df.to_csv('data.csv')
#np.savetxt(latency, newline='')   
