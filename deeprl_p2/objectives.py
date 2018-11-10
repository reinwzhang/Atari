#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:08:25 2018

@author: rein9
"""

'''Loss Function: calculate target network to predictive network difference'''
import keras.backend as K

def huber_loss(y_true, y_pred, max_grad = 1.0):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    diff = y_true - y_pred
    diff_squared = diff * diff
    max_grad_squared = max_grad * max_grad
    return max_grad_squared * (K.sqrt(1. + diff_squared/max_grad_squared) - 1.)

def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    loss = huber_loss(y_true, y_pred, max_grad=max_grad)
    return K.mean(loss, axis=None)