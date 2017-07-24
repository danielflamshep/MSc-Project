from __future__ import absolute_import
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from optimizers import adam
from mlp import forward, mse, early_stopping, gen_arch, gen_trains, list_archs
import plotting
from loading import DataLoader
from util import to_date, to_string_nn


class BayesianNeuralNet:
    def __init__(self, nn_arch, input_var, save_dir,
                 lr=0.02, iterations=500, scale=1.0, seed=15,
                 activation=None,  plot=plotting.plot_pblh):
        self.plot = plot
        self.Trained = False
        self.nn_arch = nn_arch
        self.save_dir = save_dir
        rbf = lambda x: np.exp(-x**2)
        self.activation = rbf if activation is None else activation
        self.input_var = '+'.join(a for a in input_var)
        self.shapes = list(zip(nn_arch[:-1], nn_arch[1:]))
        self.num_weights = sum((m+1)*n for m,n in self.shapes)
        rs = npr.RandomState(0)
        init_mean = rs.randn(self.num_weights)
        init_log_std = -5 * np.ones(self.num_weights)
        init_var_params = np.concatenate([init_mean, init_log_std])


    def _unpack_layers(self, weights):
        num = len(weights)
        for m, n in self.shapes:
            yield weights[:, :m*n]     .reshape((num, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num, 1, n))
            weights = weights[:, (m+1)*n:]

    def _forward(self, weights, x):
        """dim(w) = n_weight_samples x n_weights, dim(x)=(n_data x D)"""
        h = np.expand_dims(x, 0)
        for W, b in self._unpack_layers(weights):
            a = np.einsum('mnd,mdo->mno', h, W) + b
            h = self.activation(a)
        return h

    def train(self, x_train, y_train, x_test, y_test,
              optimizer=adam, loss=mse, plot_during=True):