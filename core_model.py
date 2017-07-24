from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.optimizers import adam
from plotting import plot_pblh as plot
from loading import DataLoader
from util import to_date


def train(scale_val=1.0, arch=[6,150,150,1], step=0.001, iters=1000,
          save_param=False, early_stopping=True):

    def initialize_params(scale, layer_sizes, rs=npr.RandomState(1)):
        return [(rs.randn(insize, outsize) * scale,   # weight matrix
                 rs.randn(outsize) * scale)           # bias vector
                for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

    def inference(params, inputs, nonlinearity=np.tanh):
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def mse(t, p): return 0.5 * np.sum(5.0 * (t - p) ** 2 / t.shape[0])

    def loss(weights, t):
        predictions = inference(weights, inputs)
        return 0.5 * np.sum(5.0 * (targets - predictions) ** 2 / len(inputs))

    init_params = initialize_params(scale=scale_val, layer_sizes=arch)

    d = DataLoader(LAT=44.75, LON=-80.3125, LVL=1, version=1)

    train = [[2013, 6, 7]]; test = [[2014, 7]]

    inputs, targets, inputs_test, targets_test = d.load_data(train,test)
    early_stopping=True
    # #loadV2(LAT=44.75, LON=-80.3125, LVL=1)

    file = 'train' + to_date(train) + 'test' + to_date(test) + 'NN_' + str(arch)
    #                   'LAT' + str(LAT) + 'LON' + str(LON) + 'LVL' + str(LVL) +\

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)
    plt.show(block=False)
    mse_test_list = []

    def callback(params, t, g):
        p_train = inference(params, inputs)
        p_test = inference(params, inputs_test)
        print("Iteration {} Train MSE {}".format(t, loss(params, t)))
        print('Test MSE:', mse(targets_test, p_test))
        plot(targets, targets_test, ax, bx, p_train, p_test, train, test, t)
        mse_test_list.append(mse(targets_test, p_test))

        # EARLY PLOTTING
        if early_stopping:
            if t in [200, 500, 1500] and all([m < mse_test_list[t] for m in mse_test_list[t-10:t]]):
                plt.savefig('es'+file + '.jpg')
                exit()

        if t == iters-1:
            plt.savefig(file + '.jpg')

    op_params = adam(grad(loss), init_params, step_size=step, num_iters=iters, callback=callback)

    if save_param:
        print('save param')
        np.save(file, op_params=op_params)

#train(iters=2000)