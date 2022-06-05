import torch.nn as nn
import torch
import numpy as np
import torchvision.models as models
from time import time
class Serving():
    def __init__(self, DAG, model, nodes_name):

        #DAG is the typo graph from model._trace_graph(data)
        #request_list records the info of each query the structure of the list is
        #list[0] = qeury ID
        #list[1] = arriving time
        #list[2] = inference start
        #list[3] = inference end
        #list[4] = tensor_ID
        #list[5] = query input (tensor)
        #list[6] = query output (tensor)
        self.model = model
        self.names = nodes_name
        self.request_list = []
        self.DAG = DAG
        pass
    def get_graph(self, DAG):
        self.DAG = DAG
    def merge(self, tensor, tensor_input):
        if(tensor_input == []):
            return tensor
        output = [tensor, tensor_input]
        return torch.cat(output, dim=0)

    def split(self, tensor, output_ID_list):
        stay_list = []
        tensor_output = []
        tensor_stay = []
        for i in range(0, tensor.size()[0]):
            if i not in output_ID_list:
                stay_list.append(i)
        if(len(output_ID_list) > 0):
            tensor_output = tensor[output_ID_list, :]
        else:
            tensor_output = []
        if(len(stay_list) > 0):
            tensor_stay = tensor[stay_list, :]
        else:
            tensor_stay = []
        return tensor_stay, tensor_output
    

model = models.resnet50()
data = torch.rand(1, 3, 224, 224)
DAG, DAG_layer_name = model._trace_graph(data)
model = model.to('cuda:0')
data1 = torch.rand(1, 3, 224, 224)
data2 = torch.rand(1, 3, 224, 224)
data3 = torch.rand(1, 64, 112, 112)
'''
branch1x1 = torch.rand(1, 192, 35, 35)
branch5x5 = torch.rand(1, 192, 35, 35)
branch3x3dbl = torch.rand(1, 96, 35, 35)
branch_pool = torch.rand(1, 32, 35, 35)

branch1x1 = branch1x1.to('cuda:0')
branch5x5 = branch5x5.to('cuda:0')
branch3x3dbl = branch3x3dbl.to('cuda:0')
branch_pool = branch_pool.to('cuda:0')

branch1x1_1 = torch.rand(1, 192, 35, 35)
branch5x5_1 = torch.rand(1, 192, 35, 35)
branch3x3dbl_1 = torch.rand(1, 96, 35, 35)
branch_pool_1 = torch.rand(1, 32, 35, 35)
branch1x1_1 = branch1x1_1.to('cuda:0')
branch5x5_1 = branch5x5_1.to('cuda:0')
branch3x3dbl_1 = branch3x3dbl_1.to('cuda:0')
branch_pool_1 = branch_pool_1.to('cuda:0')

data1 = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
data2 = [branch1x1_1, branch5x5_1, branch3x3dbl_1, branch_pool_1]
'''

data1 = data1.to('cuda:0')
data2 = data2.to('cuda:0')
data3 = data3.to('cuda:0')
serving = model.init_serving()
#serving.prepare_data(0, 312, data1, 0)
#serving.prepare_data(0, 174, data3, 0)
serving.prepare_data(0, 173, data2, 0)
serving.prepare_data(2, 173, data3, 1)
#serving.prepare_data(16, 174, data1, 1, cat_start = 16, cat_start_index = [17, 20, 34, 38])
#serving.prepare_data(16, 174, data2, 1, cat_start = 16, cat_start_index = [17, 20, 34, 38])
start = time()
torch.cuda.synchronize()
#with torch.no_grad():
#    out = serving(data1, start = 0, end = 30)
with torch.no_grad():
    out = serving()
torch.cuda.synchronize()
end = time()
print(end - start)
for out_tensor in out:
    print(out_tensor.size())
