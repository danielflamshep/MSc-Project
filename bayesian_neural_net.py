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
                 L2_reg = 0,noise_variance=0.1,
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
        self.init_var_params = np.concatenate([init_mean, init_log_std])
        self.rs = rs
        self.L2_reg = L2_reg
        self.noise_variance = noise_variance

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

    def logprob(self, weights, inputs, targets):
        log_prior = -self.L2_reg * np.sum(weights**2, axis=1)
        preds = self._forward(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / self.noise_variance
        return log_prior + log_lik

    @staticmethod
    def _unpack_params(params, D):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    @staticmethod
    def gaussian_entropy(log_std, D):
        return 0.5 * D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

    def variational_objective(self, logprob, params, t):
        """stochastic estimate of VLB"""
        mean, log_std = self._unpack_params(params)
        samples = self.rs.randn(self.n_samples, self.D) * np.exp(log_std) + mean
        lower_bound = self.gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound

    @staticmethod
    def _black_box_variational_inference(logprob, D, num_samples):
        """Implements http://arxiv.org/abs/1401.0118, and uses the
        local reparameterization trick from http://arxiv.org/abs/1506.02557"""

        def unpack_params(params, D):
            # Variational dist is a diagonal Gaussian.
            mean, log_std = params[:D], params[D:]
            return mean, log_std

        def gaussian_entropy(log_std):
            return 0.5 * D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

        rs = npr.RandomState(0)

        def variational_objective(params, t):
            """stochastic estimate of VLB"""
            mean, log_std = unpack_params(params)
            samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
            lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
            return -lower_bound

        gradient = grad(variational_objective)

        return variational_objective, gradient, unpack_params

    def train(self, x_train, y_train, x_test, y_test, n_samples=3, optimizer=adam):
        rs = self.rs
        gaussian_entropy = self.gaussian_entropy
        unpack_params = self._unpack_params
        predictions = self._forward
        D = self.num_weights
        logprob = self.logprob
        log_posterior = lambda weights, t: logprob(weights, x_train, y_train)

        def variational_objective(params, t):
            """stochastic estimate of VLB"""
            mean, log_std = unpack_params(params)
            samples = rs.randn(n_samples, D) * np.exp(log_std) + mean
            lower_bound = gaussian_entropy(log_std, D) + np.mean(log_posterior(samples, t))
            return -lower_bound

        objective = variational_objective

        fig, ax, bx = plotting.set_up()

        def callback(params, t, g):
            print("Iteration {} lower bound {}".format(t, -objective(params, t)))

            # Sample functions from posterior.

            rs = npr.RandomState(0)
            mean, log_std = unpack_params(params)
            sample_weights = rs.randn(2, D) * np.exp(log_std) + mean
            outputs_train = predictions(sample_weights, x_train)
            outputs_test = predictions(sample_weights, x_test)
            p_train = outputs_train[:, :, 0].T
            p_test = outputs_test[:, :, 0].T
            plotting.plot_pblh(y_train, y_test, p_train, p_test, t, ax, bx)

        var_params = optimizer(grad(objective), self.init_var_params,
                              step_size=0.01, num_iters=1000,
                              callback=callback)