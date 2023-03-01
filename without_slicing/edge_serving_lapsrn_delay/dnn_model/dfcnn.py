#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import random

import torch
import torch.nn as nn
from time import time

class View(nn.Module):
	def __init__(self):
		super(View, self).__init__()
	def forward(self, x, dim1, dim2):
		return x.view(-1, dim1, dim2)
		
class DFCNN(nn.Module): # 语音模型类
	def __init__(self, vocab_size=1000, input_dimension=200):
		'''
		初始化
		默认输出的拼音的表示大小是1434，即1433个拼音+1个空白块
		'''
		super(DFCNN,self).__init__()
		self.vocab_size = vocab_size
		self.input_dimension = input_dimension
		self.conv1 = nn.Sequential()
		self.conv1.add_module("conv1",nn.Conv2d(1, 32,kernel_size=3, padding=1))
		self.conv1.add_module("norm1",nn.BatchNorm2d(32))
		self.conv1.add_module('relu1', nn.ReLU())
		self.conv1.add_module("conv2", nn.Conv2d(32, 32, kernel_size=3, padding=1))
		self.conv1.add_module("norm2",nn.BatchNorm2d(32))
		self.conv1.add_module('relu2', nn.ReLU())
		self.conv1.add_module("pool1", nn.MaxPool2d(kernel_size = 2))

		self.conv2 = nn.Sequential()
		self.conv2.add_module("conv1",nn.Conv2d(32, 64,kernel_size=3, padding=1))
		self.conv2.add_module("norm1",nn.BatchNorm2d(64))
		self.conv2.add_module('relu1', nn.ReLU())
		self.conv2.add_module("conv2",nn.Conv2d(64, 64, kernel_size=3, padding=1))
		self.conv2.add_module("norm2",nn.BatchNorm2d(64))
		self.conv2.add_module('relu2', nn.ReLU())
		self.conv2.add_module("pool1",nn.MaxPool2d(kernel_size = 2))

		self.conv3 = nn.Sequential()
		self.conv3.add_module("conv1",nn.Conv2d(64, 128,kernel_size=3, padding=1))
		self.conv3.add_module("norm1",nn.BatchNorm2d(128))
		self.conv3.add_module('relu1', nn.ReLU())
		self.conv3.add_module("conv2",nn.Conv2d(128, 128, kernel_size=3, padding=1))
		self.conv3.add_module("norm2",nn.BatchNorm2d(128))
		self.conv3.add_module('relu2', nn.ReLU())
		self.conv3.add_module("pool1",nn.MaxPool2d(kernel_size = 2))

		self.conv4 = nn.Sequential()
		self.conv4.add_module("conv1",nn.Conv2d(128, 128,kernel_size=3, padding=1))
		self.conv4.add_module("norm1",nn.BatchNorm2d(128))
		self.conv4.add_module('relu1', nn.ReLU())
		self.conv4.add_module("conv2",nn.Conv2d(128, 128, kernel_size=3, padding=1))
		self.conv4.add_module("norm2",nn.BatchNorm2d(128))
		self.conv4.add_module('relu2', nn.ReLU())
		

		self.flatten = View()
		self.fc_features = int(self.input_dimension / 8 * 128)
		self.fc17 = nn.Linear(self.fc_features, 128, bias=False)
		self.fc18 = nn.Linear(128, self.vocab_size, bias=False)
		self.activation = nn.Softmax(dim=-1)
	def forward(self,x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.flatten(out, out.shape[2], self.fc_features)
		out = self.fc17(out)
		out = self.fc18(out)
		out = self.activation(out)
		return out