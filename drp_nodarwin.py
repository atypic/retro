#!/usr/bin/env python

from __future__ import print_function, division

from darwin.torchhelper import *
import glob

from datetime import datetime

import subprocess
import random
import time

import json
import copy

import ray
from skimage.measure import block_reduce
from scipy.misc import imsave

import gym
import gym_rle

#from bokeh.plotting import figure 
#from bokeh.io import export_png
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import xrange

import os

import psutil

import numpy as np

#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    

#ray.init(redirect_worker_output=True)
ray.init()


class Model(nn.Module):
    def __init__(self, conv_depth, output_size, img_w, img_h, maxpool_1_kernel_size = 8, img_depth=1):
        super(Model, self).__init__()

        #1 depth in, 10 kernels of 3x3 
        self.conv1 = nn.Conv2d(img_depth, 10, 3, padding=1)

        self.maxpool_1_kernel_size = maxpool_1_kernel_size

        #hiddus 
        self.rnn1 = nn.RNN(10*int(img_w/maxpool_1_kernel_size)*int(img_h/maxpool_1_kernel_size), output_size, 1, nonlinearity='relu')

        #output is supposed to be... (N, C_out, H_out, W_out)
        #10 kernels, with pooling in between
        #self.fc1 = nn.Linear(10*int(img_w/8)*int(img_h/8), output_size)

        self.h0 = Variable(torch.zeros(1,1, output_size), requires_grad=False)

    def forward(self, x):
       x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=self.maxpool_1_kernel_size, stride=self.maxpool_1_kernel_size))

       r = x.view(1,1, self.num_flat_features(x))
       
       x, self.h0 = self.rnn1(r, self.h0)

       self.h0 = self.h0.detach()

       #flattening it is important. no flatten means no worky work.
       #x = x.view(-1, self.num_flat_features(x))   #flatten it
       #fcx = self.fc1(x)
       return x

    def update_weights(self, w):
        self.weight.conv1.data = w

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features
   

@ray.remote 
class SF2Worker(object):
    def __init__(self):

        import gym_rle
        self.env = gym.make('StreetFighterIi-v0')

        self.action_size = len(self.env.env.get_action_meanings())

        #action plus down-scaled grayscale pixels
        #images from the game is (3 x 224 x 256), but we 
        #downscale to (1 x 28 x 32).

        self.model = Model(10, self.action_size, 84, 84)

        self.total_length = 0
        for p in self.model.parameters():
            kern = 1
            for s in p.size():
                kern *= s
            print("kern size ", kern)
            self.total_length += kern 
         

    def get_total_length(self):
        return self.total_length

    def obs_proc(self,obs):
        paddy = np.pad(obs[:,:252,:], ((0,28),(0,0),(0,0)), 'constant', constant_values=0)  #252,252,3
        return block_reduce(paddy, block_size=(3,3,3), func=np.max) #
        #br = obs
        #pad zeros in first dimension (height)

    def _how_to_evaluate(self, mutant):
        o = self.env.reset()
        self.env.render()
        d = False
        reward = 0.0 
        
        #darwin-to-pytorch parameters
        vector_i = 0
        for tensor, index in module_parameter_indices(self.model):
            tensor[index] = mutant[vector_i]
            vector_i += 1
        #now model is updated 

        frame = 0 

        action_distribution = {}
        action_index = {}
        for i, a in enumerate(self.env.env.get_action_meanings()):
            action_distribution[a] = 0
            action_index[i] = a

        while not d:
            o = self.obs_proc(o)  #now it's 84x84x1
            o = np.moveaxis(o, 2, 0)  #now it's 1x84x84
            o = o[np.newaxis, :]  #now it's 1x1x84x84
            o = np.asarray(o, dtype='float32') 
            o /= 255.   #make into floats

            action_pred = self.model(Variable(torch.from_numpy(o), requires_grad=False))

            action = np.argmax(action_pred.data.numpy())
          
            #uggh... 
            action_distribution[action_index[action]] += 1

            o, r, d, i = self.env.step(int(action))
            reward += r
        return reward

class Evo():
    def __init__(self):
        self.actors = [SF2Worker.remote() for _ in xrange(2)]
        self.best_solution = ray.get(self.actors[0].get_total_length.remote()) * [random.random()]
        print("Evo started")

    def do_rollout(self): 
        print("Doing rollout.")
        fitnesses = []
        rollout_ids = [actor._how_to_evaluate.remote(self.best_solution) for actor in self.actors]
        for res in ray.get(rollout_ids):
            fitnesses.append(res)
        return fitnesses


e = Evo()
for i in range(1000):
    print(e.do_rollout())
