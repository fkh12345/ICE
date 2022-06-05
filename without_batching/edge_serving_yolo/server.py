
import grpc
import time
from concurrent import futures
import inference_pb2 as data_pb2
import inference_pb2_grpc as data_pb2_grpc

from serving import Serving

import io
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import argparse

from dnn_model.lapsrn import Net
from dnn_model.dfcnn import DFCNN
from dnn_model.yolo import Yolov3
#from runtime import change_waiting_queue
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--qos', type=float, default=0.05)
args = parser.parse_args()

#model = models.resnet50()
#model = DFCNN(1000, 200)
model = Yolov3()
data = torch.rand(1, 3, 416, 416)
#data = torch.rand(1, 1, 400, 200)
#DAG, DAG_layer_name = model._trace_graph(data)

no_batching_server = Serving(model)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '8080'


class FormatData(data_pb2_grpc.FormatDataServicer):
    def DoFormat(self, request, context):
        start1 = time.time()
        str = request.text
        start = request.start
        end = request.end
        buffer = io.BytesIO(str)
        buffer.seek(0)
        input = torch.load(buffer)
        
        launch = start1
        index = no_batching_server.push_index(input, start, end)
        out, duration = no_batching_server.get_result(index)
        #print(out.shape)
        out_time = time.time()
        
        return data_pb2.actionresponse(text=out_time - start1, queue=duration)  
 
 
def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=5000))  
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(), grpcServer)  
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)    
    grpcServer.start()
    try:
        no_batching_server()
    except KeyboardInterrupt:
        grpcServer.stop(0) 
 
 
if __name__ == '__main__':
    serve()