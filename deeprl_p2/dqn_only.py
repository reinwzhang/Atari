#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:08:16 2018

@author: rein9
"""
# =============================================================================
# to import nessisary packages
import numpy as np
import sys
import keras.models
import keras.backend as K
from deeprl_p2.utils import *
# =============================================================================
class DQNAgent:
    """Class implementing DQN.

	This is a basic outline of the functions/parameters you will need in order to implement the DQNAgnet.

	Parameters
	----------
	q_network: keras.models.Model; Your Q-network model.
	preprocessor: deeprl_p2.core.Preprocessor; The preprocessor class.
	memory: deeprl_p2.core.Memory; Your replay memory.
	gamma: float; Discount factor.
	target_update_freq: float;	  Frequency to update the target network.
        can be a number representing a soft target update (see utils.py)
        or a hard target update (see utils.py and Atari paper.)
    num_burn_in: The number of samples to fill in replay memory before begin updating the Q-network
    train_freq: How often to actually update your Q-Network. For every Q-Nerwork update,
        to check if the stability is improved by collecting a couple samples for your replay memory
    batch_size: How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 q_values_func,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 save_path):
        
        self.q_network = q_network
        # copy the current network as the target network
        self.target_network = keras.models.clone_model(q_network)
        self.target_network.set_weights(q_network.get_weights())
        #input[input_state,input_action]--> conv --> conv --> flatten --> dense --> output:q_values
        # q_v = K.function([input_state], [q_values])
        self.target_q_values_func = K.function([self.target_network.layers[0].input], [self.target_network.layers[5].output])
        self.q_values_func = q_values_func
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.save_path = save_path
        self.num_steps = 0
        # can be train, test or init
        self.mode = 'init'

    def compile(self, optimizer, loss_func):
        '''
        Optimizer:Adam, Loss: mean_huber_loss
        '''
        self.q_network.compile(optimizer = optimizer, loss = loss_func)
        self.target_network.compile(optimizer = optimizer, loss = loss_func)

    def load_weights(self, weights_path):
        self.q_network.load_weights(weights_path)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def calc_q_values(self, state):
        '''To get q_values from state'''
        return self.q_values_func([state])[0]

    def calc_target_q_values(self, state):
        return self.target_q_values_func([state])[0]

    def select_action(self, state, **kwargs):
        '''Based of current state to select actions
        1. Init --> If you are still collecting random sample--> RandomnPolicy
        2. Testing --> GreedyEpsilonPolicy with a low epsilon
        3. Training --> LinearDecayGreedyEpsilonPolicy
        Hepler func: process_state_for_network
        return: action'''
        preprocessed_state = self.preprocessor.process_state_for_network(state)
        q_values = self.calc_q_values(preprocessed_state)
        return self.policy[self.mode].select_action(q_values), preprocessed_state

    def update_predict_network(self):
        """Update your predict network.
        Behavior may differ based on what stage of training your in.
        1. training mode: check if you should update your network parameters based on the current step and the value you set for train_freq.
            1) sample a minibatch --> 2) calculate the target values
            --> 3) update your network,--> 4) then update your target values.
        @return: the loss and other metrics as an output
        """
        states, actions, rewards, new_states, is_terminals = self.memory.sample(self.batch_size)
        preprocessed_states, preprocessed_new_states = self.preprocessor.process_batch(states, new_states)
        actions = self.preprocessor.process_action(actions)
        #update the nerwork
        q_values = self.calc_target_q_values(preprocessed_new_states)
        # get the max q_values
        max_q_values = np.max(q_values,axis = 1)
        max_q_values[is_terminals] = 0.0
        targets = rewards + self.gamma * max_q_values
        targets = np.expand_dims(targets, axis = 1) #need to expand the dimension to be (size,)

        self.q_network.train_on_batch([preprocessed_states, actions], targets)

        if self.num_steps % self.target_update_freq == 0:
            print('Update the Target Network at %d steps'% self.num_steps)
            self.update_target_network()

    def fit(self, env, num_iterations, max_episode_length = None):
        '''
        Parameters
        env: gym env; This is the Atari Environment. Wrap the env in wrap_atati_env function in utils.py
        num_iterations: Number of iterations to train
        max_episode_length:How a single episode lasts before the agent rests.
        '''
        print('Initializing replay memory...')
        sys.stdout.flush()
        self.mode = 'init'
        self.memory.clear()
        self.preprocessor.reset()
        self.num_steps = 0
        num_updates = 0
        num_episodes = 0
        while num_updates < num_iterations:
            state = env.reset()
            self.preprocessor.reset()
            num_episodes += 1
            t = 0
            total_reward = 0
            while True:
                self.num_steps += 1
                t += 1
                action, _ = self.select_action(state)
#                action, internal_state = self.select_action(state)
                next_state, reward, is_terminal, debuf_info = env.step(action)
                #get the reward
                reward = self.preprocessor.process_reward(reward)
                total_reward += reward
                #get the next state and D array to memory
                preprocessed_state = self.preprocessor.process_state_for_memory(state)
                self.memory.append(preprocessed_state, action, reward, is_terminal)

                if self.num_steps > self.num_burn_in:
                    if self.mode != 'train':
                        print('Finish Burn-in, start Training...')
                    self.mode = 'train'
                    if self.num_steps % self.train_freq == 0:
                        print('Training... Update Q-Network...')
                        self.update_predict_network()
                        num_updates += 1
                        if num_updates % 10000 == 0:
                            print('Save model for at %d num_updates' % num_updates)
                            self.q_network.save_weights('%s/model_weights_%d.h5' % (self.save_path, num_updates //10000))
                if is_terminal or (max_episode_length is not None and t > max_episode_length):
                    break
                state = next_state
            print('Episode %d ends, lasted for %d steps (total %d steps), total reward is: %d. (%d/%d updates)' % (num_episodes, t, self.num_steps, total_reward, num_updates, num_iterations))

    def evaluate(self, env, num_episodes, max_episode_length = None):
        '''Test the agent with provided env
        No updating network step in this function
        @return: np.mean(rewards), np.std(rewards), average_episode_length / num_episodes
        '''
        self.mode = 'test'
        average_episode_length = 0.
        rewards = []

        for i in range(num_episodes):
            state = env.reset()
            t = 0
            episode_reward = 0.0
            while True:
                t += 1
                action, _ = self.select_action(state)
                next_state, reward, is_terminal, debug_info = env.step(action)
                # unlike training, we dont need to preprocess the reward in testing
                episode_reward += reward
                average_episode_length += 1

                if is_terminal or (max_episode_length is not None and t > max_episode_length):
                    break
                state = next_state
            rewards.append(episode_reward)
        self.mode = 'train'
        return np.mean(rewards), np.std(rewards), average_episode_length / num_episodes
