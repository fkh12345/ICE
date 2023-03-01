
import grpc
import time
from concurrent import futures
import inference_pb2 as data_pb2
import inference_pb2_grpc as data_pb2_grpc

from multiprocessing import Semaphore

import io
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import argparse

from dnn_model.lapsrn import Net
from dnn_model.dfcnn import DFCNN
from dnn_model.yolo import Yolov3
from runtime import change_waiting_queue

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--qos', type=float, default=0.05)
parser.add_argument('--method', type=str, default="ICE")
parser.add_argument('--progress', type=str, default="true")
args = parser.parse_args()

#model = models.inception_v3()
model = Net()
#model = DFCNN(1000, 200)
#model = Yolov3()
DAG, DAG_layer_name = model._trace_graph()
model.set_profile(True)
model.set_input([0, 39])
model.set_profile(False)
model_input = model.get_input()
model = model.to('cuda:0')

serving = model.init_serving()

se = Semaphore(1)
isprocess = False
print('init server is ready')

serving_window = Semaphore(args.bs)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '8080'

class FormatData(data_pb2_grpc.FormatDataServicer):
    def DoFormat(self, request, context):
        start1 = time.time()
        str = request.text
        start = request.start
        end = request.end

        index = serving.push_data(start, str, [0, 39], "cuda:0")

        serving_window.acquire()
        se.acquire()
        #while(isprocess is not False):
        #    time.sleep(0.001)
            #print('waiting')
        
        if(start > 0):
            serving.push_queue(start1, 0.1, 0.1)
        else:
            serving.push_queue(start1, 0.1, 0.1)
        query_id = serving.get_index()

        serving.prepare_data(start, end, input, query_id, index)
        actual_index = serving.ID_map[query_id]
        
        #print("query", query_id, "finish input")
        se.release()
        #torch.cuda.synchronize()
        #while(not serving.have_result[query_id]):
            #time.sleep(0.001)
        while(True):
            if(start == 0):
                a = 1
            actual_index = serving.ID_map[query_id]
            if(serving.have_result[actual_index] == True):
                break
            else:
                time.sleep(0.001)
            
        launch = serving.launch
       
        output = serving.output[actual_index]
        out_time = serving.out_time[actual_index]
        
        #print(out_time, start1, out_time - start1)
        #print('client: ', ' ', query_id, end2, end1, end2 - end1)
        serving.return_result(query_id)
        
        serving_window.release()
        #print(end1 - start1)
        #print(serving.start)
        return data_pb2.actionresponse(text=out_time - start1, queue=out_time - launch)  
 
 
def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=10000))  
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(), grpcServer)  
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)    
    grpcServer.start()
    try:
        while True:
            time.sleep(0.005)
            #se.acquire()
            #print('change to True')
            #se.release()
            now = time.time()
            if(len(serving.input) == 0):
                isprocess = False
                #print('change to False')
                continue
            if(args.method == 'ICE'):
                serving.queue_time = change_waiting_queue(serving.start, serving.queue_time_origin, [0, 39])
            if(len(serving.input) >= args.bs):
                se.acquire()
                isprocess = True
                query_input = len(serving.input)
                #print(query_input, serving.start.count(0), query_input - serving.start.count(0))
                start2 = time.time()
                with torch.no_grad():
                    out_bs = serving()
                torch.cuda.synchronize()
                end2 = time.time()
                
                #print(1, query_input, out_bs, start2, end2)
                
                #for out_tensor in out:
                #    print(out_tensor.size())
                #print(len(serving.input))
                se.release()
            elif(now > min(serving.queue_time)):
                se.acquire()
                q_time = serving.q_time_np
                sss = serving.start_np
                pos = np.where(q_time < now)
                minpos = np.argmin(q_time)
                q_type = 0
                if(args.method != 'ICE' or args.progress != 'true'):
                    q_type = sss[minpos]
                #pos = np.where(q_time < now)
                #minpos = np.argmin(q_time)
                #sss = np.array(serving.start)
                #print(sss[pos], sss[minpos], q_time[pos] - now)
                query_input = len(serving.input)
                #print(query_input, serving.start.count(0), query_input - serving.start.count(0))
                #isprocess = True
                start2 = time.time()
                with torch.no_grad():
                    out_bs = serving(q_type)
                torch.cuda.synchronize()
                end2 = time.time()
                #print(2, query_input, out_bs, start2, end2)
                
                #for out_tensor in out:
                #    print(out_tensor.size())
                se.release()
            #se.acquire()
            isprocess = False
            #print('change to False')
            #se.release()
            
            #time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0) 
 
 
if __name__ == '__main__':
    serve()