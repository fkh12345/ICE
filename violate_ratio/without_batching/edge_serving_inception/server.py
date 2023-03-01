
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
parser.add_argument('--worker', type=int, default=0)
args = parser.parse_args()

model = models.inception_v3()
#model = DFCNN(1000, 200)
#model = Yolov3()
model.set_profile(True)
model.set_input([0, 63])
model.set_profile(False)
model_input = model.get_input()


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '808{}'.format(args.worker)
device = 'cuda:{}'.format(args.worker)


no_batching_server = Serving(model, device)
class FormatData(data_pb2_grpc.FormatDataServicer):
    def DoFormat(self, request, context):
        start1 = time.time()
        str = request.text
        start = request.start
        end = request.end
        index1 = model.push_data(start, str, [0, 63], "cpu")
        launch = start1
        index = no_batching_server.push_index(input, start, end, index1)
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