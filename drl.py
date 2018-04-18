#!/usr/bin/env python

from __future__ import print_function, division

from datetime import datetime

import subprocess

import json
import copy

import ray
from skimage.measure import block_reduce
from scipy.misc import imsave

import gym
import gym_rle

import darwin.evolution as dre
from darwin.rl import AgentRunner
from darwin.rl.normalizer import Normalizer
from darwin.rl.rlproblem import RLProblem, spawn_actor
from darwin.randomhelper import Randomness
from darwin.repeatedvalue import RepeatedValue

import pickle
import cloudpickle

from six.moves import xrange

from sacred import Experiment
from darwin.sacredhelper import DontExperiment

import os

import psutil

import numpy as np


if __name__ == "__main__":
    ex = Experiment()
else:
    ex = DontExperiment()


@ex.config
def config():
    nactors = -1  # how many ray actors will be created (if <0, set as #CPUs)
    envname = 'StreetFigherIi-v0'  # which gym environment is to be solved
    optimizer = 'ARS'  # which search method is to use (ARS,XNES,SNES,BDNES)
    niters = 10000  # number of iterations
    popsize = 20  # population size
    truncation_size = 20  # truncation size for ARS
    learning_rate = 0.1  # NES learning rate for search distribution reshaping
    center_learning_rate = 1.0  # NES learning rate for the center of dist
    stdev = 0.1  # standard deviation of the search distribution
    nsamples = 10  # by using how many trajectories will an agent be evaluated
    same_seed = True  # will the trajectories be created using the same seed
    alive_bonus_to_remove = -1.0  # If >0, this amount will be removed from R(t)
    observation_normalization = False  # "virtual batch normalization"


ray.init()

class SF2(dre.BaseProblem):
    def __init__(self, solution_vector_length):

        dre.BaseProblem.__init__(
            self,
            objective_sense = dre.ObjectiveSense.maximization,
            initial_lower_bounds = solution_vector_length * [0.],
            initial_upper_bounds = solution_vector_length * [0.01])

        self.env = gym.make('StreetFighterIi-v0')

        self.vis = False
        self.current_iter = -1
    
    def obs_proc(self,obs):
        return block_reduce(obs, block_size=(8,8,3), func=np.max)

    def _how_to_evaluate(self, x):
        o = self.env.reset()
        d = False
        reward = 0.0 
        
        x = np.array(x, dtype=float)
        x = x.reshape((-1,20))

        last_action_vector = np.zeros((20))
        frame = 0 
        while not d:
            #action = env.action_space.sample()
            action = np.matmul(
                        np.append(
                            self.obs_proc(o).flatten(),
                            last_action_vector),
                    x)
            last_action_vector = copy.copy(action)

            action = np.argmax(action)
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
            subprocess.run(['rm', '-rf', '/tmp/iter*.png'])
            subprocess.run(['scp',
                            '/tmp/vid' + str(self.current_iter) + '.mp4',
                            'atypic@anipsyche.net:/var/www/anipsyche.net/streetfighter/'])
            
        return reward

    SolutionVector = dre.BaseProblem.RealValuedSolutionVector 




@ex.command
def visualize(fname, iteration):

    with open(fname, 'r') as fp:
        sol = json.loads(fp.read())
  
    length = int((20 + (224/8) * (256/8)) * 20)
    foo = SF2(length)
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
#        runner,
#        seed=0 if same_seed else None,
#        with_observation_normalization=observation_normalization)

    length = int((20 + (224/8) * (256/8)) * 20)
    problem = SF2(length)

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


