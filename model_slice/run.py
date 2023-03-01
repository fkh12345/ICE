import numpy as np
import pandas as pd
import os

def run_baseline(model, net, device):
    layer = []
    os.chdir('dnn_slice/')
    cmd = 'python graph_cut.py --dnn {} --net {}  --mobile {}'.format(model, net, device)
    result = os.popen(cmd)
    tmp = result.readline()
    r1 = float(tmp.split(' ')[2])
    rs = tmp.split(' ')[0]
    re = tmp.split(' ')[1]
    cmd1 = 'python graph_baseline.py --dnn {} --net {}  --mobile {}'.format(model, net, device)
    result = os.popen(cmd1)
    tmp = result.readline()
    r2 = float(tmp.split(' ')[0])
    r3 = float(tmp.split(' ')[1])
    if(device == 'kirin'):
        layer = [rs, re]
    os.chdir('..')
    return r1, r2, r3, layer

r_ice = []
r_neu = []
r_db = []
r_l = []
r_r = []
n_l = []
d_l = []
for device in ['kirin', 'mi', 'pi']:
    for net in [4, 5]:
        tmp_ice = [] 
        tmp_neu = []
        tmp_db = []
        tmp_l = []
        tmp_r = []
        tmp_nl = []
        tmp_dl = []
        for i, benchmark in enumerate(['resnet', 'vgg', 'inception', 'lapsrn', 'dfcnn', 'yolo', 'bert']):
            tmp1, tmp2, tmp3, tmp4 = run_baseline(benchmark, net, device)
            tmp_ice.append(tmp1)
            tmp_neu.append(tmp2)
            tmp_db.append(tmp3)
            if(tmp4 != []):
                if(float(tmp4[0]) < 0.05):
                    tmp4[0] = 0
                tmp_l.append(tmp4[0])
                tmp_r.append(tmp4[1])
                tmp_nl.append(tmp2)
                tmp_dl.append(tmp3)
        r_ice.append(tmp_ice)
        r_neu.append(tmp_neu)
        r_db.append(tmp_db)
        if(tmp_l != []):
            r_l.append(tmp_l)
            r_r.append(tmp_r)
            n_l.append(tmp_nl)
            d_l.append(tmp_dl)


r_ice = np.average(np.array(r_ice), 0)
r_neu = np.average(np.array(r_neu), 0)
r_db = np.average(np.array(r_db), 0)
r_l = np.array(r_l)
r_r = np.array(r_r)
n_l = np.array(n_l)
d_l = np.array(d_l)
result = np.array([r_ice, r_neu, r_db])
result = np.concatenate([result, r_l, r_r, n_l, d_l], 0).T
df = pd.DataFrame(result)
df.to_csv('data.csv')