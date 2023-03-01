import csv
import math
import numpy as np
import copy
import argparse
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--dnn', type=str, default='lapsrn')
parser.add_argument('--net', type=float, default=5)
parser.add_argument('--mobile', type=str, default='kirin')
args = parser.parse_args()

dnn = args.dnn
mobile = args.mobile
folder = 'dag/'




start = time()

file = dnn + '_data.csv'
file = folder + file
f = open(file ,'r')
csv_reader = csv.reader(f)
DAG_profile_d1 = []
DAG_profile_d2 = []
DAG_profile_c = []
for i, line in enumerate(csv_reader):
    if (i > 0):
        DAG_profile_d1.append(float(line[6]))
        if(args.mobile == 'kirin'):
            DAG_profile_d2.append(float(line[1]))
        elif(args.mobile == 'mi'):
            DAG_profile_d2.append(float(line[2]))
        elif(args.mobile == 'pi'):
            DAG_profile_d2.append(float(line[3]))
        if(args.net == 5):
            DAG_profile_c.append(float(line[4]))
        elif(args.net == 4):
            DAG_profile_c.append(float(line[5]))

DAG_profile_d1 = np.array(DAG_profile_d1)
DAG_profile_d2 = np.array(DAG_profile_d2)
DAG_profile_c = np.array(DAG_profile_c)

length = len(DAG_profile_d1)
time = 100000000
cut = -1
for slice in range(0, length):
    t_d1 = np.sum(DAG_profile_d1[slice:length])
    t_d2 = np.sum(DAG_profile_d2[0:slice])
    t_c = DAG_profile_c[slice]
    t_e2e = t_d1 + t_d2 + t_c + DAG_profile_c[-1]
    if(t_e2e < time):
        cut = slice
        time = t_e2e
trans_start = DAG_profile_c[0]
cloud_d = np.sum(DAG_profile_d1)
edge_d = np.sum(DAG_profile_d2)
device = 1
if(edge_d < cloud_d + trans_start):
    device = 0
print(1-cut/length, device)


