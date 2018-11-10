#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:25:02 2018

@author: rein9
"""
# =============================================================================
import argparse, os, random, sys, gym
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from gym import wrappers
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv2D, Dense, Flatten, Input, Lambda
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import deeprl_p2 as tfrl
from deeprl_p2.dqn import DQNAgent
from deeprl_p2.objectives import mean_huber_loss
from deeprl_p2.preprocessors import PreprocessorSequence
from deeprl_p2.policy import UniformRandomPolicy, GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_p2.core import ReplayMemory
# =============================================================================

def create_model(window, input_shape, num_actions, dueling_type = 'avg', model_name = 'q_network'):
    '''
    window: defines how many frames are in the sequence
    input_shape: tuple(int, int), the expected input image size
    num_actions: Num of possible actions
    '''
    network_model = None
    if model_name == 'q_network':
        with tf.name_scope('q_network'):
            with tf.name_scope('input'):
                input_state = Input(shape = (window, input_shape[0], input_shape[1]))
                input_action = Input(shape = (num_actions, ))
            with tf.name_scope('conv1'):
                conv1 = Conv2D(16,(8,8), data_format='channels_first', kernel_initializer = 'glorot_uniform', activation='relu', padding = 'valid', strides = (4,4))(input_state)
            with tf.name_scope('conv2'):
                conv2 = Conv2D(32,(4,4), data_format='channels_first', kernel_initializer = 'glorot_uniform', activation='relu', padding = 'valid', strides = (2,2))(conv1)
            with tf.name_scope('fc'):
                flattened = Flatten()(conv2)
                dense1 = Dense(256, kernel_initializer = 'glorot_uniform', activation='relu')(flattened)
            with tf.name_scope('output'):
                q_values = Dense(num_actions, kernel_initializer='glorot_uniform', activation=None)(dense1)
                q_v = dot([q_values, input_action], axes=1)
            network_model = Model(inputs=[input_state, input_action], outputs=q_v)
            q_values_func = K.function([input_state], [q_values])
        print('q_network summary')
        network_model.summary()
    elif model_name == 'double_q_network':
        pass
    elif model_name == 'dueling_q_network':
        with tf.name_scope('q_network'):
            with tf.name_scope('input'):
                input_state = Input(shape = (window, input_shape[0], input_shape[1]))
                input_action = Input(shape = (num_actions, ))
            with tf.name_scope('conv1'):
                conv1 = Conv2D(16,(8,8), data_format='channels_first', kernel_initializer = 'glorot_uniform', activation='relu', padding = 'valid', strides = (4,4))(input_state)
            with tf.name_scope('conv2'):
                conv2 = Conv2D(32,(4,4), data_format='channels_first', kernel_initializer = 'glorot_uniform', activation='relu', padding = 'valid', strides = (2,2))(conv1)
            with tf.name_scope('fc'):
                flattened = Flatten()(conv2)
                dense1 = Dense(256, kernel_initializer = 'glorot_uniform', activation='relu')(flattened)
            with tf.name_scope('output'):
#                q_values = Dense(num_actions, kernel_initializer='glorot_uniform', activation=None)(dense1)
                # layer y has a shape (nb_action+1,)
                # y[:,0] represents V(s;theta)
                # y[:,1:] represents A(s,a;theta)
                y = Dense(num_actions + 1, activation='linear')(dense1)
                # caculate the Q(s,a;theta)
                # dueling_type == 'avg'
                # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
                # dueling_type == 'max'
                # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
                # dueling_type == 'naive'
                # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
                if dueling_type == 'avg':
                    q_values = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(num_actions,))(y)
                elif dueling_type == 'max':
                    q_values = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(num_actions,))(y)
                elif dueling_type == 'naive':
                    q_values = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(num_actions,))(y)
                else:
                    assert False, "dueling_type must be one of {'avg','max','naive'}"
                q_v = dot([q_values, input_action], axes=1)
            network_model = Model(inputs=[input_state, input_action], outputs=q_v)
            q_values_func = K.function([input_state], [q_values])
        print('q_network summary')
        network_model.summary()
    else:
        print('model not implemented')
        sys.exit(0)
    return network_model, q_values_func

def get_output_folder(parent_dir, env_name):
    '''
    return save folder
    '''
    os.makedirs(parent_dir, exist_ok = True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1]) # to get the last saved output folder
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1
    
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok = True)
    return parent_dir

def train(args):
    # need to start a new session
    env = gym.make(args.env)
    num_actions = env.action_space.n
    if args.modeltype == 'dueling_dqn':
        network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, args.dueling_type, model_name ='dueling_q_network')
    else:
        network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, args.dueling_type, model_name ='q_network')
    preprocessor = PreprocessorSequence(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions)
    memory = ReplayMemory(args.memsize, args.stack_frames)
    policy = {
                'init':    UniformRandomPolicy(num_actions),
                'train':   GreedyEpsilonPolicy(num_actions),
                'test':    GreedyPolicy(),
    }

    print("Generate Model...")
    if args.modeltype == 'double_dqn':
        ddqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, True, False, args.dueling_type)
        dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, False, False, args.dueling_type)
    elif args.modeltype == 'dueling_dqn':
        dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, False, True, args.dueling_type)
    else: # natual dqn
        dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, False, False, args.dueling_type)

    print("Compiling Model...")
    dqn_agent.compile(optimizer=Adam(lr = args.learning_rate), loss_func = mean_huber_loss)
    if args.modeltype == 'double_dqn':
        ddqn_agent.compile(optimizer=Adam(lr = args.learning_rate), loss_func = mean_huber_loss)

    print("Fitting model...")
    sys.stdout.flush()
    dqn_rewards = dqn_agent.fit(env, args.num_iterations, args.max_episode_length)
    if args.modeltype == 'double_dqn':
        ddqn_rewards = ddqn_agent.fit(env, args.num_iterations, args.max_episode_length)
        return ddqn_rewards
    return dqn_rewards

def test(args):
    if not os.path.isfile(args.model_path):
        print("The model path: {} doesn't exist in the system.".format(args.model_path))
        print("Hints: python dqn_atari.py --mode test --modeltype dueling_dqn --dueling_type avg --model_path Path_to_your_model_weigths")
        return

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    env = gym.make(args.env)
    num_actions = env.action_space.n
    if args.modeltype == 'dueling_dqn':
        network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, args.dueling_type, model_name ='dueling_q_network')
    else:
        network_model, q_values_func = create_model(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions, args.dueling_type, model_name ='q_network')
    rewards = []
    lens = []
    tries = 0
    while True:
        env = gym.make(args.env)
        env = wrappers.Monitor(env, 'videos', force=True)


        preprocessor = PreprocessorSequence(args.stack_frames, (args.cropped_size, args.cropped_size), num_actions)
        memory = ReplayMemory(args.memsize, args.stack_frames)
        policy = {
                  'init':   UniformRandomPolicy(num_actions),
                  'train':  GreedyEpsilonPolicy(num_actions),
                  'test':   GreedyPolicy(),
        }
        if args.modeltype == 'double_dqn':
            dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, True, False, args.dueling_type)
        elif args.modeltype == 'dueling_dqn':
            dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, False, True, args.dueling_type)
        else: # natual dqn
            dqn_agent = DQNAgent(network_model, q_values_func, preprocessor, memory, policy, args.gamma, args.target_update_freq, args.num_burn_in, args.train_freq, args.batch_size, args.output, False, False, args.dueling_type)
        #network_model.load_weights(args.output + '/%s_model_weights_%d.h5' % (args.modeltype, m))
        dqn_agent.load_weights(args.model_path)
        cumulative_reward, std, average_episode_length = dqn_agent.evaluate(env, 1, None)
        tries += 1

        # Sometime the model is not very stable.
        if tries > 100 or cumulative_reward > 350:
            break

        print ('average reward = %f, std = %f, average_epis_length = %d' % (cumulative_reward, std, average_episode_length))
        rewards.append(cumulative_reward)
        lens.append(average_episode_length)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--output', default='model-weights', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--memsize', default=1000000, type=int, help='Replay Memory Size')
    parser.add_argument('--mode', default='train', type=str, help='Train or test')
    parser.add_argument('--stack_frames', default=4, type=int, help='The number of stacked frames')
    parser.add_argument('--cropped_size', default=84, type=int, help='The size of the cropped windows')
    parser.add_argument('--max_episode_length', default=10000, type=int, help='the maximum of episode to be ran')
    parser.add_argument('--gamma', default=0.99, type=float, help='The reward discount parameter')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='how many steps to update target network')
    parser.add_argument('--num_burn_in', default=12000, type=int, help='how many frames to burn in the memory before traiing')
    parser.add_argument('--train_freq', default=10, type=int, help='How often you actually update your Q-Network. Sometimes stability is improved if you collect a couple samples for your replay memory, for every Q-network update that you run.')
    parser.add_argument('--batch_size', default=32, type=int, help='size of each training batch')
    parser.add_argument('--learning_rate', default=0.00025, type=float, help='size of each training batch')
    parser.add_argument('--num_iterations', default=2000000, type=int, help='the number of iteration to run')
    parser.add_argument('--modeltype', default='dqn', type=str, help='enable Double dqn: double_dqn; dueling_dqn: dueling_dqn')
    parser.add_argument('--dueling_type', default='avg', type=str, help='dueling type for dueling_dqn')
    parser.add_argument('--model_path', default='', type=str, help='path to the model')

    args = parser.parse_args()

    if args.mode == "train":
        args.output = get_output_folder(args.output, args.env)
        train_rewards = train(args)
        print(train_rewards)
    elif args.mode == "test":
        test_rewards = test(args)
        print(test_rewards)
#        plt.plot(np.array(test_rewards), c='b', label='double')
#        plt.legend(loc='best')
#        plt.ylabel('Q eval')
#        plt.xlabel('training steps')
#        plt.grid()
#        plt.show()
