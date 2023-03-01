from concurrent.futures import thread
import threading
import torch
import time
import torch.nn as nn
xxx = threading.Condition()
class Serving():
    def __init__(self, model, device):
        self.data_queue = []
        self.cv = []
        self.result_queue = []
        self.start = []
        self.end = []
        self.duration = []
        self.queue_cv = threading.Condition()
        self.device = "cuda:0"
        self.finished = []
        self.model = model.to(self.device)
        print('server have been started!')
    
    def __call__(self):
        while(True):
            for i, cv in enumerate(self.cv):
                if(self.data_queue[i] == None or self.finished[i] == True):
                    continue

                time.sleep(0.005)
                self.cv[i].acquire()
                #print('start %d query' %(i))
                input_data = []
                input_index = []
                self.model.set_start_end(self.start[i], self.end[i])
                max_query = 14
                for num, q_start in enumerate(self.start):
                    if(len(input_data) >= max_query):
                        break
                    if(self.data_queue[num] == None):
                        continue
                    if(q_start == self.start[i]):
                        input_data.append(self.data_queue[num])
                        input_index.append(num)
                        if(num != i):
                            self.cv[num].acquire()
                input_data = torch.cat(input_data, 0)
                #print(input_data.shape[0])
                input_data = input_data.to(self.device)
                start = time.time()
                with torch.no_grad():
                    out = self.model(input_data)
                out = out.cpu()
                end = time.time() - start
                #self.data_queue[i] = self.data_queue[i].to(self.device)
                #with torch.no_grad():
                #    self.result_queue[i] = self.model(self.data_queue[i])
                #print("result %d return" %(i))
                #self.result_queue[i] = self.result_queue[i].cpu()
                for item in input_index:
                    self.result_queue[item] = torch.tensor([1])
                    if(item != i):
                        self.data_queue[item] = None
                        self.finished[item] = True
                        self.duration[item] = end
                        self.cv[item].notify()
                        self.cv[item].release()
                        
                self.data_queue[i] = None
                self.finished[i] = True
                self.duration[item] = end
                self.cv[i].notify()
                self.cv[i].release()
                break
                
    
    def push_index(self, data, start, end, index1):
        self.queue_cv.acquire()
        index = len(self.data_queue)
        self.data_queue.append(nn.Module.query_input[index1 - 1])
        self.result_queue.append(None)
        self.start.append(start)
        self.end.append(end)
        self.duration.append(-1)
        self.finished.append(False)
        self.cv.append(threading.Condition())
        self.queue_cv.notify()
        self.queue_cv.release()
        return index
    
    def get_result(self, index):
        self.cv[index].acquire()
        while(not isinstance(self.result_queue[index], torch.Tensor)):
            self.cv[index].wait()
        self.cv[index].notify()
        self.cv[index].release()
        self.cv[index] = None
        self.result_queue[index] = None
        self.start[index] = -1
        self.end[index] = -1
        self.data_queue[index] = None
        result = self.result_queue[index]
        duration = self.duration[index]
        return result, duration

