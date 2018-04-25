#!/usr/bin/env python

from __future__ import print_function, division

import glob

from datetime import datetime

import subprocess

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


import darwin.evolution as dre
from darwin.rl import AgentRunner
from darwin.rl.normalizer import Normalizer
from darwin.rl.rlproblem import RLProblem, spawn_actor
from darwin.randomhelper import Randomness
from darwin.repeatedvalue import RepeatedValue
from darwin.torchhelper import *

import pickle
import cloudpickle

from six.moves import xrange

from sacred import Experiment
from darwin.sacredhelper import DontExperiment

import os

import psutil

import numpy as np

#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    

if __name__ == "__main__":
    ray.init(redis_address=ray.services.get_node_ip_address() + ":6379")
    ex = Experiment()
else:
    ex = DontExperiment()


@ex.config
def config():
    nactors = 150  # how many ray actors will be created (if <0, set as #CPUs)
    envname = 'StreetFigherIi-v0'  # which gym environment is to be solved
    optimizer = 'ARS'  # which search method is to use (ARS,XNES,SNES,BDNES)
    niters = 10000  # number of iterations
    popsize = 60  # population size
    truncation_size = 20  # truncation size for ARS
    learning_rate = 0.05  # NES learning rate for search distribution reshaping
    center_learning_rate = 1.0  # NES learning rate for the center of dist
    stdev = 0.1  # standard deviation of the search distribution
    nsamples = 10  # by using how many trajectories will an agent be evaluated
    same_seed = True  # will the trajectories be created using the same seed
    alive_bonus_to_remove = -1.0  # If >0, this amount will be removed from R(t)
    observation_normalization = False  # "virtual batch normalization"



class Model(nn.Module):
    def __init__(self, conv_depth, output_size, img_w, img_h, img_depth=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)


        #hiddus 
        self.rnn1 = nn.RNN(10*int(img_w/8)*int(img_h/8), output_size, 1, nonlinearity='relu')

        #output is supposed to be... (N, C_out, H_out, W_out)
        #10 kernels, with pooling in between
        #self.fc1 = nn.Linear(10*int(img_w/8)*int(img_h/8), output_size)

        self.h0 = Variable(torch.zeros(1,1, output_size), requires_grad=False)

    def forward(self, x):
       x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=8, stride=8))

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
 
class SF2(dre.BaseProblem):
    def __init__(self):

        
        self.env = gym.make('StreetFighterIi-v0')


        self.action_size = len(self.env.env.get_action_meanings())
        print("Action size: ", self.action_size)
        print(self.env.env.get_action_meanings())

        #action plus down-scaled grayscale pixels
        #images from the game is (3 x 224 x 256), but we 
        #downscale to (1 x 28 x 32).

        self.model = Model(10, self.action_size, 256, 256)

        total_length = 0
        for p in self.model.parameters():
            kern = 1
            for s in p.size():
                kern *= s
            print("kern size ", kern)
            total_length += kern 
         
        print("total ", total_length)

        #self.solution_vector_length = self.model.para
        
        self.solution_vector_length = total_length
        print("Solution vector length given to darwin: ", self.solution_vector_length)

        dre.BaseProblem.__init__(
            self,
            objective_sense = dre.ObjectiveSense.maximization,
            initial_lower_bounds = self.solution_vector_length * [-1.0],
            initial_upper_bounds = self.solution_vector_length * [1.0])


        self.vis = False
    
    def obs_proc(self,obs):
        br = block_reduce(obs, block_size=(1,1,3), func=np.max)
        #br = obs
        #pad zeros in first dimension (height)
        return np.pad(br, ((0,32),(0,0),(0,0)), 'constant', constant_values=0)

    def _how_to_evaluate(self, mutant):
        o = self.env.reset()
        d = False
        reward = 0.0 
             
        #params = torch.from_numpy(np.array(params, dtype=float))
     
        #from mutant to torch.
          
          
        #x = x.reshape((-1,self.))

        last_action_vector = np.zeros((self.action_size))

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
            o = self.obs_proc(o)  #now it's 28x32x1
            o = np.moveaxis(o, 2, 0)  #now it's 1x28x32
            o = o[np.newaxis, :]  #now it's 1x1x3x3
            o = np.asarray(o, dtype='float32')
            o /= 255.

            action_pred = self.model(Variable(torch.from_numpy(o), requires_grad=False))

            action = np.argmax(action_pred.data.numpy())
          
            #uggh... 
            action_distribution[action_index[action]] += 1

            o, r, d, i = self.env.step(int(action))
            reward += r
            if self.vis:
                #TODO: Dump to .png's
                arr = self.env.render(mode='rgb_array')
                imsave('/tmp/iter_'+str(self.current_iter)+'_'+str(frame)+'.png', arr)
            frame += 1

        if self.vis:
            
            
            subprocess.run(['ffmpeg',
                            '-y',
                            '-i', 
                            '/tmp/iter_' + str(self.current_iter) + '_%d.png',
                            '-c:v',
                            'libx264', 
                            '-vf', 'fps=50', '-pix_fmt', 'yuv420p',
                            '/tmp/vid' + str(self.current_iter) + '.mp4'])

            for i in glob.glob('/tmp/*iter*.png'):
                subprocess.run(['rm', i])

            subprocess.run(['scp',
                            '/tmp/vid' + str(self.current_iter) + '.mp4',
                            'atypic@anipsyche.net:/var/www/anipsyche.net/streetfighter/best.mp4'])

            #fitness plot
            reward_history=None
            with open('rewards.log', 'r+') as fp:
                loaded = json.loads(fp.read())
                loaded['reward_log'].append(reward)
                reward_history = loaded['reward_log'] 

            with open('rewards.log', 'w+') as fp:
                print(json.dumps({'reward_log': reward_history}), file=fp)

            fig, ax  = plt.subplots(nrows = 1, ncols = 1)
            ax.plot(reward_history)
            fig.savefig('/tmp/reward_plot.png')
            plt.close(fig)
            
            #metadata json
            with open('/tmp/evometa.json', 'w') as fp:
                print(json.dumps(
                    {'iteration': self.current_iter,
                     'actions': action_distribution,
                     'reward_log' : reward_history,
                     'reward': reward}
                    ), file=fp)


            subprocess.run(['scp',
                            '/tmp/evometa.json',
                            '/tmp/reward_plot.png',
                            'atypic@anipsyche.net:/var/www/anipsyche.net/streetfighter/'])
           
            for i in glob.glob('/tmp/*vid*.mp4'):
                subprocess.run(['rm', i])

           
        return reward

    SolutionVector = dre.BaseProblem.RealValuedSolutionVector 




@ex.command
def visualize(fname, iteration):

    with open(fname, 'r') as fp:
        sol = json.loads(fp.read())
  
    foo = SF2()
    foo.vis = True
    foo.current_iter = iteration

    foo._how_to_evaluate(np.array(sol, dtype=float))


@ex.automain
def main(nactors,
         envname,
         optimizer,
         niters,
         popsize,
         truncation_size,
         learning_rate,
         center_learning_rate,
         stdev,
         nsamples,
         same_seed,
         alive_bonus_to_remove,
         observation_normalization,
#         visualize_at_end,
         _seed):


    	
    
    with open('rewards.log', 'w') as fp:
        print(json.dumps({'reward_log': list()}), file=fp)
    # get the class of the search method
    # e.g.: if the configuration `optimizer` is given as "XNES",
    #       the variable `search` stores the class `dre.XNES`
    search = getattr(dre, optimizer)

    # determine how many ray actors to be created
    # if not specified, then the number of ray actors is the CPU count
    if nactors < 0:
        nactors = psutil.cpu_count()

    # generate a name for the log file name
#   log_fname = '{}_{}_{}_{}.hdf5'.format(
#       datetime.now().strftime('%Y%m%d%H%M%S'),
#       envname,
#       optimizer,
#       str(os.getpid()))

    # do we want the reward(t) at every time step t
    # to be decreased by a certain amount?
    if alive_bonus_to_remove <= 0.0:
        alive_bonus_to_remove = None

    # initialize the random number generator
    rndgen = Randomness(_seed)

    # initialize the agent runner
#   runner = AgentRunner(environment_name=envname,
#                        scenario_samples_count=nsamples,
#                        alive_bonus_to_remove=alive_bonus_to_remove)

    # initialize the problem description
#    problem = RLProblem(
#        runnerm,
#        seed=0 if same_seed else None,
#        with_observation_normalization=observation_normalization)

    problem = SF2()

    # initialize the Parameters of the search method
    params = search.Parameters()

    # initialize the ray actors
    actors = [spawn_actor(problem) for _ in xrange(nactors)]
    params.ray_actors = actors

    # set the population size
    params.population_size = popsize

    # set the truncation size
    if issubclass(search, dre.ARS):
        params.truncation_size = truncation_size

    # set the block sizes for BDNES
    if issubclass(search, dre.BDNES):
        params.block_sizes = runner.get_block_sizes_for_bdnes()

    # set the stdev
    if issubclass(search, dre.XNES):
        sollength = runner.get_solution_vector_length()
        params.initial_stdev = np.array([float(stdev)] * sollength)
    else:
        params.stdev = float(stdev)

    # set the learning rates
    params.learning_rate = learning_rate
    if issubclass(search, dre.XNES):
        params.center_learning_rate = center_learning_rate

    # initialize the search algorithm
    searcher = search(problem, rndgen, params)

    # start the optimization
    for iteration in xrange(1, niters + 1):
        searcher.iterate()

        # if the observation normalization configuration is on,
        # then we have to communicate with the workers
        if observation_normalization:
            # collect observation stats from the actors
            collected_stats = ray.get(
                [actor.get_collected_observation_stats.remote()
                 for actor in actors])

            # update the normalizer of the master problem definition
            for stats in collected_stats:
                problem.update_observation_normalizer(stats)

            # upload the updated master normalizer to the actors
            ray.get(
                [actor.use_master_observation_normalizer.remote(
                    problem.get_collected_observation_stats())
                 for actor in actors])

        ex.log_scalar('best_solution', searcher.best.evaluation)
        ex.log_scalar('center_solution', searcher.population_center.evaluation)

        log_fname = 'sol_' + str(iteration) + '.json'

        with open(log_fname, 'w') as fp:
            print(json.dumps(list(searcher.best.get_values())), file=fp)

        visualize(log_fname, iteration)

        #log(log_fname, agentrunner=runner, optimizer=searcher)

        print("Iteration:", iteration)
        #print("Best solution:", searcher.best)
        print("Center solution's cumulative reward:",
              searcher.population_center.evaluation)
        print("  Best solution's cumulative reward:",
              searcher.best.evaluation)

    print("Adding the artifact...")
    ex.add_artifact(log_fname)
    print("Added!")


