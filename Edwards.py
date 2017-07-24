#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).
Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.
References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from loading import DataLoader
import plotting


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(X) + np.random.normal(0, noise_std, size=N)
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  print(X.shape, y.shape)
  return X, y


train_mns = [[2014, 6]]
test_mns = [[2014, 7]]
dl = DataLoader(scale='none', ignore=None, grad_height=5)
input_vars = dl.height_grad_vars
x_train, y_train, x_test, y_test = dl.load_data(train_mns, test_mns, input_vars=input_vars)
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
N = x_train.shape[0]
D = x_train.shape[1]


def neural_network(X):
  h = tf.tanh(tf.matmul(X, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return tf.reshape(h, [-1])


ed.set_seed(42)

dim_0 = 25
dim_1 = 10

with tf.name_scope("model"):
  W_0 = Normal(loc=tf.zeros([D, dim_0]), scale=tf.ones([D, dim_0]), name="W_0")
  W_1 = Normal(loc=tf.zeros([dim_0, dim_1]), scale=tf.ones([dim_0, dim_1]), name="W_1")
  W_2 = Normal(loc=tf.zeros([dim_1, 1]), scale=tf.ones([dim_1, 1]), name="W_2")
  b_0 = Normal(loc=tf.zeros(dim_0), scale=tf.ones(dim_0), name="b_0")
  b_1 = Normal(loc=tf.zeros(dim_1), scale=tf.ones(dim_1), name="b_1")
  b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

  X = tf.placeholder(tf.float32, [N, D], name="X")
  y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(X.get_shape()[0]), name="y")

# INFERENCE
with tf.name_scope("posterior"):
  with tf.name_scope("qW_0"):
    qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, dim_0]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([D, dim_0]), name="scale")))
  with tf.name_scope("qW_1"):
    qW_1 = Normal(loc=tf.Variable(tf.random_normal([dim_0, dim_1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([dim_0, dim_1]), name="scale")))
  with tf.name_scope("qW_2"):
    qW_2 = Normal(loc=tf.Variable(tf.random_normal([dim_1, 1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([dim_1, 1]), name="scale")))
  with tf.name_scope("qb_0"):
    qb_0 = Normal(loc=tf.Variable(tf.random_normal([dim_0]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([dim_0]), name="scale")))
  with tf.name_scope("qb_1"):
    qb_1 = Normal(loc=tf.Variable(tf.random_normal([dim_1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([dim_1]), name="scale")))
  with tf.name_scope("qb_2"):
    qb_2 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([1]), name="scale")))

inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={X: x_train, y: y_train})

inference.run(logdir='log', n_iter=10000)
sess = ed.get_session()
p_train = sess.run(y, feed_dict={X: x_train})
p_test = sess.run(y, feed_dict={X: x_test[:x_train.shape[0]]})
print(p_test.shape, p_train.shape)
t = None
pad = np.abs(x_test.shape[0]-x_train.shape[0])
p_test = np.lib.pad(p_test, (0, pad), 'constant')
fig, ax, bx = plotting.set_up()
plotting.plot_pblh(y_train, y_test, p_train, p_test, t, ax, bx)


