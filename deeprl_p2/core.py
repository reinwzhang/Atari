#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:37:41 2018

@author: rein9
"""

# =============================================================================
import numpy as np
# =============================================================================

class Sample:
    '''Represents a RL sample: a standard (s,a,r,s',terminal) tuple
    '''
    pass

class Preprocessor:
    '''Base class for preprocessor
    Preprocessors are implemented as class so that they can have internal state.
    This can be usefule for things like AtariPreprocessors which maxes over k frames
    '''
    def preprocess_state_for_network(self, state):
        '''This should be called before the action is selected,
        This is different from preprocess_state_for_memory, 'uint8' works better for storage
        but neural network works for with floatint point images'''
        return state
    
    def preprocess_state_for_memory(self, state):
        return state
    
    def preprocess_batch(self, samples):
        return samples
    
    def preprocess_reward(self, reward):
        '''For clipping reward if the reward is too big or too small'''
        return reward
    
    def reset(self):
        pass

class ReplayMemory:
    ''' Interface for replay memories. This is very imporant to save memory
    '''
    
    def __init__(self, max_size, window_length):
        '''
        max_size: max memory allocated
        window_length: two pointer 
        '''
        
        self.psize = 84 # we pick image size of 84*84 for this project
        self.max_size = max_size
        self.window_length = window_length
        self.mem_size = (max_size + window_length - 1)
        self.mem_state = np.ones((self.mem_size, self.psize, self.psize), dtype = np.uint8)
        self.mem_action = np.ones(self.mem_size, dtype = np.int8)
        self.mem_reward = np.ones(self.mem_size, dtype = np.float32)
        self.mem_terminal = np.ones(self.mem_size, dtype = np.bool)
        self.start = 0
        # end points to the next position; The content doesnt change when end points at it;
        # but will change when end points move forward
        self.end = 0
        self.full = False
    
    def append(self, state, action, reward, is_terminal):
        if self.start == 0 and self.end == 0: # the initial frame
            # Init: 1 2 3 S E
            for i in range(self.window_length - 1):
                # need to repeat the first frame by window size
                self.mem_state[i] = state
                self.start = (self.start + 1) % self.mem_size
            self.mem_state[self.start] = state
            self.mem_action[self.start] = action
            self.mem_reward[self.start] = reward
            self.mem_terminal[self.start] = is_terminal
            self.end = (self.start + 1) % self.mem_size
        else:
            # Case 1:  1 2 3 S ... E
            # Case 2:  ... E 1 2 3 S ...            
            self.mem_state[self.end] = state
            self.mem_action[self.end] = action
            self.mem_reward[self.end] = reward
            self.mem_terminal[self.end] = is_terminal
            self.end = (self.end + 1) % self.mem_size
            if self.end > 0 and self.end < self.start:
                self.full = True
            if self.full:
                self.start = (self.start + 1) % self.mem_size
                
    def sample(self, batch_size, indexes = None):
        '''
        return batch_size samples from memory starting from indexes
        '''
        if self.end == 0 and self.start == 0:
            # memory is empty
            # state, action, reward, next_state, is_terminal
            return None, None, None, None, None
        else:
            count = 0
            if self.end > self.start:
                # memory is not full yet
                count = self.end - self.start
            else:
                count = self.max_size
            
            if count <= batch_size:
                indices = np.arrange(0, count - 1)
            else:
                # random sample batch_size of samples from memory
                indices = np.random.randint(0, count-1, size = batch_size)
            
            # the following is some manual calculation because we set "4" as our current state frame number
            # state: K-3 ~ K(the core frame)
            # next state: k-2 ~ K+1
            indices_5 = (self.start + indices + 1) % self.mem_size
            indices_4 = (self.start + indices ) % self.mem_size
            indices_3 = (self.start + indices - 1) % self.mem_size
            indices_2 = (self.start + indices - 2) % self.mem_size
            indices_1 = (self.start + indices - 3) % self.mem_size            
            frame_5 = self.mem_state[indices_5]
            frame_4 = self.mem_state[indices_4]
            frame_3 = self.mem_state[indices_3]
            frame_2 = self.mem_state[indices_2]
            frame_1 = self.mem_state[indices_1]
# =============================================================================
#                 x = np.ones((1, 2, 3))
#                 np.transpose(x, (1, 0, 2)).shape --> (2, 1, 3)
#            state_list = np.array([frame_1, frame_2, frame_3, frame_4])
#            state_list = np.transpose(state_list, [1,0,2,3])
#            --> state_list = [frame_2, frame_1, frame_3, frame_4]
# =============================================================================

            state_list = np.array([frame_1, frame_2, frame_3, frame_4])
            state_list = np.transpose(state_list, [1,0,2,3])
            
            next_state_list = np.array([frame_2, frame_3, frame_4, frame_5])
            next_state_list = np.transpose(next_state_list, [1,0,2,3])
            
            action_list = self.mem_action[indices_4]
            reward_list = self.mem_reward[indices_4]
            terminal_list = self.mem_terminal[indices_4]
            
            return state_list, action_list, reward_list, next_state_list, terminal_list
        
    def clear(self):
        self.start = 0
        self.end = 0
        self.full = False