#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:08:33 2018

@author: rein9
"""
import semver
import tensorflow as tf
from six.moves import cPickle

def get_uninitialized_variables(variables=None):
    '''Return a list of unitilized tf variables
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.get_default_session(config = config)
    if variables is None:
        variables =tf.global_variables()
    else:
        variables = list(variables)
        
    if len(variables) ==0:
        return []
    
    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]

def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    pass

def load_pk(path):
    fin = open(path, 'rb')
    obj = cPickle.load(fin)
    fin.close()
    return obj

def save_as_pk(data, filename):
    fout = open(filename, 'wb')
    cPickle.dump(data, fout, protocol = cPickle.HIGHEST_PROTOCOL)
    fout.close()
