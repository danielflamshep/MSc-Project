from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.util import flatten
from autograd.optimizers import adam
from getdataV2 import loadV2
from numpy import arange as ar

LAT, LON, LVL = 44.7, -80.5, 1
def init_random_params(scale, layer_sizes, rs=npr.RandomState(10)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def nn_predict(params, inputs, nonlinearity=np.tanh):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs


def log_gaussian(params, scale):
    flat_params, _ = flatten(params)
    return np.sum(norm.logpdf(flat_params, 0, scale))


def logprob(weights, inputs, targets, noise_scale=0.1):
    predictions = nn_predict(weights, inputs)
    return np.sum(norm.logpdf(predictions, targets, noise_scale))

if __name__ == '__main__':

    init_params = init_random_params(scale=0.1, layer_sizes=[5, 10,25,10,1])
    inputs, targets, inputs_test, targets_test = loadV2(LAT=44.75, LON=-80.3125, LVL=1)

    def objective(weights, t):
        return -logprob(weights, inputs, targets)\
               #-log_gaussian(weights, scale=10.0)

    def loss(weights, t):
        predictions = nn_predict(weights, inputs)
        scale=5.0
        return 0.5*np.sum(scale*(targets-predictions)**2 /len(targets))


    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(211, frameon=False)
    bx = fig.add_subplot(212, frameon=False)
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} loss {}".format(t, loss(params, t)))

        # Plot data and functions
        ax.cla()
        bx.cla()

        # PLOT TRAINING TIME SERIES
        plot_inputs = ar(len(targets))       # TIME SERIES INPUTS
        ax.plot(plot_inputs, targets, 'b')

        outputs = nn_predict(params, inputs)
        ax.plot(plot_inputs, outputs, 'g', ms=1)
        ax.set_ylim([0, 2])
        ax.set_ylabel('PBLH in km')
        ax.set_title('training data : JUNE 2013')

        bx.plot(ar(len(targets_test)), targets_test, 'b')
        plot_test_inputs = ar(len(targets_test))
        outputs_test = nn_predict(params, inputs_test)
        bx.plot(plot_test_inputs, outputs_test, 'r', lw=2)
        bx.set_ylim([0, 2])
        bx.set_ylabel('PBLH in km')
        bx.set_title('test data : JULY 2013')

        plt.pause(1.0/60.0)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(loss), init_params,
                            step_size=0.01, num_iters=2000, callback=callback)