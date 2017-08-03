
import matplotlib.pyplot as plt
import plotting
import autograd.numpy as np
import autograd.numpy.random as npr

from black_box_svi import black_box_variational_inference as bbvi
from autograd.optimizers import adam
from loading_vers2 import DataLoader


def construct_nn(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


if __name__ == '__main__':


    train_mns = [[2014, 9]]
    test_mns = [[2014, 10]]
    dl = DataLoader()
    ins = dl.train_vars
    x_train, y_train, x_test, y_test = dl.load_data(train_mns, test_mns, input_vars=ins)
    x_dim = x_train.shape[1]


    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)

    num_weights, predictions, logprob = construct_nn(layer_sizes=[x_dim, 36, 36, 1], L2_reg=0,
                                                     noise_variance=0.1, nonlinearity=rbf)

    log_posterior = lambda weights, t: logprob(weights,  x_train, y_train)

    objective, gradient, unpack_params = bbvi(log_posterior, num_weights, num_samples=10)

    fig, ax, bx = plotting.set_up()

    def callback(params, t, g):
        # Sample functions from posterior.

        rs = npr.RandomState(0)
        mean, log_std = unpack_params(params)
        sample_weights = rs.randn(2, num_weights) * np.exp(log_std) + mean
        outputs_train = predictions(sample_weights, x_train)
        outputs_test = predictions(sample_weights, x_test)
        p_train = outputs_train[:, :, 0].T
        p_test = outputs_test[:, :, 0].T
        plotting.plot_pblh(y_train, y_test, p_train, p_test, t, ax, bx)

        if t % 10 == 0 :
            metric = np.mean(np.abs(y_test - p_test))
            print("ITER {} MEAN PBLH DIFF {}".format(t, metric))



    rs = npr.RandomState(0)
    init_mean    = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    var_params = adam(gradient, init_var_params, step_size=0.01, num_iters=1000, callback=callback)