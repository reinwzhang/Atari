#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:08:58 2018

@author: rein9
"""
'''
def different policies for different stage of training/test
'''
# =============================================================================
import numpy as np
# =============================================================================

class Policy:
    """Base class representing an MDP policy.
    """

    def select_action(self, **kwargs):
        raise NotImplementedError('This method should be overriden.')


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.
    """
    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, q_values, **kwargs):
        """Return a random action index.
        """
        return np.random.randint(0, self.num_actions)

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}

class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.
    """
    def select_action(self, q_values, **kwargs):  # noqa: D102
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.
    With probability epsilon choose a random action. Otherwise choose the greedy action.
    """
    def __init__(self, num_actions, epsilon=0.05):
        assert num_actions >= 1
        self.num_actions = num_actions
        self.epsilon = epsilon

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.
        """
        assert self.num_actions == q_values.shape[1]

        greedy_action = np.argmax(q_values)
        if np.random.random() >= self.epsilon:
            return greedy_action

        action = np.random.randint(0, self.num_actions)
        return action



class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.
    Like GreedyEpsilonPolicy but the epsilon decays from a start value to an end value over k steps.
    """
    def __init__(self, num_actions, start_value=0.999, end_value=0.1,
                 num_steps=1000000):
        assert num_actions >= 1
        self.num_actions = num_actions
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.steps = 0.0

    def select_action(self, q_values, **kwargs):
        """Decay parameter and select action.
        """
        assert self.num_actions == q_values.shape[1]

        epsilon = self.start_value + (self.steps / self.num_steps) * (self.end_value - self.start_value)
        if self.steps < self.num_steps:
            self.steps += 1.0

        greedy_action = np.argmax(q_values)
        if np.random.random() >= epsilon:
            return greedy_action

        action = np.random.randint(0, self.num_actions)
        return action


    def reset(self):
        """Start the decay over at the start value."""
        self.steps = 0.0

class SoftmaxPolicy(Policy):
    """ Implement softmax policy for multinimial distribution
    Simple Policy
    - takes action according to the pobability distribution
    """
    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, probs):
        """Return the selected action
        # Arguments
            probs (np.ndarray) : Probabilty for each action
        # Returns
            action
        """
        action = np.random.choice(range(self.num_actions), p=probs)
        return action

class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, num_actions, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        assert num_actions >= 1
        self.num_actions = num_actions
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert self.num_actions == q_values.shape[1]
        q_values = q_values.astype('float32')
        
        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(self.num_actions), p = probs)
        return action
        
        