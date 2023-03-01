# ICE
The source code of paper on SC2022 

## Introduction
ICE is an irregular collaborative serving engine that can enable efficient inference execution in cloud-edge continuum with two main modules. The model slicer adaptive slice DNN model into pieces, and the runtime serving engine enable multi-entrance multi-exit inference to support irregular serving of model slices on the datacenter side.

## Hardware and Software Depedencies

- Hardware&software requirements

  1. Hardware Requirements

     1. CPU: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
     2. Memroy: 252G
     3. NVIDIA RTX 2080 Ti GPU

  2. Software Requirements

     1. Ubuntu 18.04
     2. Docker 19.03
     3. GPU Driver: 450.51
     4. CUDA 10.1
     5. CUDNN 7.6
     6. Miniconda3-py37_4.9.2
     7. Pytorch 1.3.0

## Preparing environment
1. Download and run the provided runtime backend with docker.
```shell
$ docker pull midway2018/ice_runtime
$ docker run -it --gpus=all --ipc=host  -it midway2018/ice_runtime /bin/bash 
$ git clone https://github.com/fkh12345/ICE.git
```
2. Activate conda and create python environment with essential dependencies
```shell
$ conda activate slice
$ cd ICE
$ pip install -r requirement.txt
$ # Switch to backend without batching policy
$ conda activate no-batch
$ pip install -r requirement.txt
```

## How to run
The running progress of the ICE runtime is seperated by two steps: 1) Start the inference server with `server.py` script. And 2) Run the emulation threads using the `client1.py` script.

### Inference with progress-aware policy

```shell
$ conda activate slice
$ python server.py --bs <DNN_batchsize> --method ICE --progress true

$ # A new terminal
$ python client1.py --bs <num_of_queries> --load <high/medium/low> --slice true
```
