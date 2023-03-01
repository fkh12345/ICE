import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import time
from tqdm import tqdm

PYTHON='/root/miniconda3/envs/slice/bin/python'

def start_server(path ,cmd):
    os.chdir(path)
    os.system(cmd)

def kill_server():
    result = os.popen('ps -ef|grep server.py|awk \'{print $2}\'|head -1')
    pid = int(result.readline())
    os.system('kill -kill ' + str(pid))
    os.system('kill -kill ' + str(pid+1))

def run_adaptiveness(method, slice):
    path = 'edge_serving_vgg_delay_ratio'
    server_cmd = '{} server.py --bs 64 --method {} --progress true > /dev/null'.format(PYTHON, method)
    p = mp.Process(target=start_server, args=(path, server_cmd))
    p.start()
    time.sleep(20)
    os.chdir(path)
    cmd = '{} client1.py --bs 2700 --slice {}'.format(PYTHON, slice)
    for i in range(0, 3):
        os.system(cmd + ' > /dev/null')
        time.sleep(1)
    result_list = []
    cmd = '{} client1.py --bs 2700 --slice {}'.format(PYTHON, slice)
    for i in tqdm(range(0, 6)):
        result = os.popen(cmd)
        result = result.readline()
        result = result[1:-2].split(', ')
        result = [float(i) for i in result]   
        result_list.append(result)
        time.sleep(1)
    
    np_result = np.array(result_list)
    np_result = np.average(np_result, 0)

    kill_server()
    os.chdir('..')
    return np_result


data_ice = run_adaptiveness('ICE', 'true')
data_db = run_adaptiveness('DB', 'false')

result = np.array([data_ice, data_db])
df = pd.DataFrame(result.T)
df.to_csv('data.csv')
