import csv
import math
import numpy as np
import copy
import argparse
from time import time
def Deep_search(DAG, now_layer, branch_deep, DAG_Output, is_skip_branch=False):

    if(now_layer == -1):
        return DAG_Output, -1
    now_node = DAG[now_layer]
    next_layer = -1
    if(now_layer == -1):
        return DAG_Output, -1
    if((DAG_Name[now_layer] == "aten::add" or DAG_Name[now_layer] == "aten::cat") and is_skip_branch == False):
        return DAG_Output, now_layer

    if(len(now_node) == 1):
        next_node = DAG[now_layer][0]
        if(DAG_Name[now_layer] != "aten::add" and DAG_Name[now_layer] != "aten::cat"):
            DAG_Output.append(now_layer)
        DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output)
    elif(len(now_node) == 0):
        DAG_Output.append(now_layer)
        DAG_Output, next_layer = Deep_search(DAG, -1, branch_deep, DAG_Output)
    elif(len(now_node) > 0): #if one layer have more than one branch this layer will be a big node
        big_node = []
        if(branch_deep > 0):
            DAG_Output.append(now_layer)
        for next_node in DAG[now_layer]:
            if(branch_deep == 0):
                branch_node = [now_layer]
                branch_node, next_layer = Deep_search(DAG, next_node, branch_deep + 1, branch_node)
                branch_node.append(next_layer)
                if(len(branch_node) > 2):
                    big_node.append(branch_node)
            else:
                DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep + 1, DAG_Output)
        next_node = next_layer
        if(branch_deep == 0):
            if(len(big_node) > 1):
                DAG_Output.append(big_node)
            else:
                for item in big_node[0]:
                    DAG_Output.append(item)
            DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output, True)
        else:
            DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output, True)
            DAG_Output.append(next_layer)

    return DAG_Output, next_layer
        

def build_execute_graph(DAG_Output, DAG_profile_d1, DAG_profile_d2, DAG_t):

    execute_graph = []
    for index, item in enumerate(DAG_Output):
        stage_graph = []
        stage_min = []
        if(isinstance(item, int)): 
            stage_graph.append(DAG_profile_d1[item])
            stage_graph.append(DAG_profile_d1[item] + DAG_t[item])
            stage_graph.append(DAG_profile_d2[item] + DAG_t[item])
            stage_graph.append(DAG_profile_d2[item])

        elif(isinstance(item, list)):
            big_node = DAG_Output[index]
            lengh = len(big_node)
            branch_pos = [0 for _ in range(lengh)]
            min_time = 10000000 
            min_pos = 0
            for top, bottom in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                min_time = 10000000
                min_pos = 0
                for num in range(int(math.pow(2, lengh))):
                    execute_time = [0, 0] 
                    trans_time = [0, 0, 0, 0] 
                    bin_num = list(bin(num))
                    while(len(bin_num) < lengh + 2):
                        bin_num.insert(2, '0')
                    for index in range(2, len(bin_num)):
                        branch_pos[index - 2] = int(bin_num[index])
                    for index, value in enumerate(branch_pos):      
                        branch = big_node[index]
                        if(value == 0):
                            execution_branch = np.array(DAG_profile_d1)[branch]
                            execution_branch = execution_branch[1:-1]
                            trans_top = DAG_t[branch[0]]
                            trans_bottom = DAG_t[branch[-2]]
                            execute_time[0] += np.sum(execution_branch)
                            if(trans_time[0] == 0 and top == 1):
                                trans_time[0] += trans_top
                            if(bottom == 1):
                                trans_time[2] += trans_bottom
                        elif(value == 1):
                            execution_branch = np.array(DAG_profile_d2)[branch]
                            execution_branch = execution_branch[1:-1]
                            trans_top = DAG_t[branch[0]]
                            trans_bottom = DAG_t[branch[-2]]
                            execute_time[1] += np.sum(execution_branch)
                            if(trans_time[1] == 0 and top == 0):
                                trans_time[1] += trans_top
                            if(bottom == 0):
                                trans_time[3] += trans_bottom
                    profile_time = np.max([execute_time[0] + trans_time[0] + trans_time[2] ,execute_time[1] + trans_time[1] + trans_time[3]])
                    if(top == 0):
                        profile_time += DAG_profile_d1[big_node[0][0]]
                    elif(top == 1):
                        profile_time += DAG_profile_d2[big_node[0][0]]
                    if min_time > profile_time:
                        min_time = profile_time
                        min_pos = num
                stage_graph.append(min_time)
                stage_min.append(min_pos)
            stage_graph.append(stage_min)
        execute_graph.append(stage_graph)    
    return execute_graph

def short_path(execute_path, wait_time):
    now_layer_d1 = 0
    now_layer_d2 = 0
    past_layer_d1 = []
    past_layer_d2 = []
    for i, item in enumerate(execute_path):
        next_d1 = [0, 0]
        next_d2 = [0, 0]
        next_d1[0] = now_layer_d1 + item[0]
        next_d1[1] = now_layer_d2 + item[2] 
        next_d2[0] = now_layer_d1 + item[1] + wait_time
        next_d2[1] = now_layer_d2 + item[3]
        if(next_d1[0] < next_d1[1]):
            now_layer_d1 = next_d1[0]
            tmp_d1 = copy.deepcopy(past_layer_d1)
            tmp_d1.append(0)

        elif(next_d1[0] >= next_d1[1]):
            now_layer_d1 = next_d1[1]
            tmp_d1 = copy.deepcopy(past_layer_d2)
            tmp_d1.append(1)

        if(next_d2[0] < next_d2[1]):
            now_layer_d2 = next_d2[0]
            tmp_d2 = copy.deepcopy(past_layer_d1)
            tmp_d2.append(0)

        elif(next_d2[0] >= next_d2[1]):
            now_layer_d2 = next_d2[1]
            tmp_d2 = copy.deepcopy(past_layer_d2)
            tmp_d2.append(1)

        past_layer_d1 = tmp_d1
        past_layer_d2 = tmp_d2
    return now_layer_d1, now_layer_d2, past_layer_d1, past_layer_d2


parser = argparse.ArgumentParser()
parser.add_argument('--dnn', type=str, default='bert')
parser.add_argument('--net', type=float, default=5)
parser.add_argument('--mobile', type=str, default='kirin')
args = parser.parse_args()

DAG = []
dnn = args.dnn
mobile = args.mobile
file = dnn + '.dag'
folder = 'dag/'
file = folder + file
f = open(file, 'r')
csv_reader = csv.reader(f)
for line in csv_reader:
    for i, item in enumerate(line):
        line[i] = int(item)
    DAG.append(line)
DAG_Name = []
file = dnn + '.name'
file = folder + file
f = open(file, 'r')
csv_reader = csv.reader(f)
for line in csv_reader:
    DAG_Name.append(line[0])
DAG_Output = []


DAG_Output, _ = Deep_search(DAG, 0, 0, DAG_Output)
tmp = len(DAG)
for item in DAG_Output:
    if(isinstance(item, list)):
        if(len(item) > 2):
            tmp = len(DAG_Output)

DAG_Output_name = []
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

cloud_time = np.sum(np.array(DAG_profile_d1)) + (DAG_profile_c[0] + DAG_profile_c[-1])
wait_time = np.sum(np.array(DAG_profile_d1)) * 0
execute_graph = build_execute_graph(DAG_Output, DAG_profile_d1, DAG_profile_d2, DAG_profile_c)
input_trans = execute_graph[0]

input_trans[0] = 1000000
input_trans[1] = 1000000
execute_graph[0] = input_trans
now_layer_d1, now_layer_d2, past_layer_d1, past_layer_d2 = short_path(execute_graph, wait_time)

d1_time = np.sum(np.array(DAG_profile_d1)) + (DAG_profile_c[0] + DAG_profile_c[-1])
wait_time = np.sum(np.array(DAG_profile_d1)) * 0
execute_graph = build_execute_graph(DAG_Output, DAG_profile_d1, DAG_profile_d2, DAG_profile_c)
input_trans = execute_graph[0]

input_trans[0] = 1000000
input_trans[1] = 1000000
execute_graph[0] = input_trans
now_layer_d1, now_layer_d2, past_layer_d1, past_layer_d2 = short_path(execute_graph, wait_time)

past_layer_d2 = np.array(past_layer_d2)
l_len = past_layer_d2.shape[0] - 1
exe_start = -1
exe_end = -1
serve_cloud = False
while(l_len >= 0):
    if(past_layer_d2[l_len] == 0 and serve_cloud == False):
        exe_end = l_len
        serve_cloud = True
    elif(past_layer_d2[l_len] == 1 and serve_cloud == True):
        exe_start = l_len
        serve_cloud = False 
        break
    l_len = l_len - 1
ratio = (exe_end - exe_start)/len(DAG_Output)
exe_start = (exe_start+1)/(tmp)
exe_end = (exe_end+1)/(tmp)

 
print(exe_start, exe_end, ratio)
for item in DAG_Output:
    if(isinstance(item, int)):
        DAG_Output_name.append(DAG_Name[item])
    else:
        tmp = []
        for i in range(len(item)):
            tmp1 = []
            for value in item[i]:
                tmp1.append(DAG_Name[value])
            tmp.append(tmp1)
        DAG_Output_name.append(tmp)
