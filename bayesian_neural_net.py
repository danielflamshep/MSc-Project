from __future__ import absolute_import
from __future__ import print_function
import os
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from optimizers import adam
import plotting
from lidar_data_loader import DataLoader
from util import to_date, arch_string, Dataholder


class BayesianNeuralNet:
    def __init__(self, nn_arch,  save_dir,input_var=None,
                 lr=0.01, iterations=1500, seed=0,
                 l2_reg=0, noise_variance=0.1,
                 plot=plotting.plot_pblh,
                 labels=None, activation=None):
        self.lr = lr
        self.plot = plot
        self.labels = labels
        self.Trained = False
        self.arch = nn_arch
        self.save_dir = save_dir
        #self.input_var = '+'.join(a for a in input_var)

        self.rs = npr.RandomState(seed)
        self.shapes = list(zip(nn_arch[:-1], nn_arch[1:]))
        self.num_weights = sum((m+1)*n for m, n in self.shapes)

        rbf = lambda x: np.exp(-x**2)
        init_mean = self.rs.randn(self.num_weights)
        init_log_std = -5 * np.ones(self.num_weights)
        self.activation = rbf if activation is None else activation
        self.init_var_params = np.concatenate([init_mean, init_log_std])

        self.L2_reg = l2_reg
        self.iters = iterations
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

    def get_preds(self, params, x_train, x_test, p_samples):
        mean, log_std = self._unpack_params(params, self.num_weights)
        sample_weights = self.rs.randn(p_samples, self.num_weights) * np.exp(log_std) + mean
        outputs_train = self._forward(sample_weights, x_train)
        outputs_test = self._forward(sample_weights, x_test)
        p_train = outputs_train[:, :, 0].T
        p_test = outputs_test[:, :, 0].T
        return p_train, p_test

    def train(self, x_train, y_train, x_test, y_test,
                    loss=mse, optimizer=adam,
                    train_samples=3, p_samples=1,
                    plot_during=False):

        rs = self.rs
        D = self.num_weights
        get_preds = self.get_preds
        plot = self.plot
        gaussian_entropy = self.gaussian_entropy
        unpack_params = self._unpack_params

        log_posterior = lambda weights, t: self.logprob(weights, x_train, y_train)

        def variational_objective(params, t):
            """stochastic estimate of VLB"""
            mean, log_std = unpack_params(params, D)
            samples = rs.randn(train_samples, D) * np.exp(log_std) + mean
            lower_bound = gaussian_entropy(log_std, D) + \
                          np.mean(log_posterior(samples, t))
            return -lower_bound

        objective = variational_objective

        d = Dataholder(y_train, y_test, loss, self.save_dir, self.arch)
        d.labels = self.labels
        fig, axes = plotting.set_up(show=False)

        def callback(params, t, g):
            # print("I {} VLB {}".format(t, -objective(params, t)))
            p_train, p_test = get_preds(params, x_train, x_test, p_samples)
            d.get_ptest(p_test)
            d.get_ptrain(p_train)

            if plot_during:
                plot(d, axes)

            #if t % 500.0 == 0:
                #print("{}| TEST MHD {:.3f}".format(t, d.MHD_test))

            if d.MHD_test < 0.15:
                d.iters = t
                plot(d, axes, draw=False, save=True)
                print('plotted with MHD: {:.3f}'.format(d.MHD_test))

        var_params = optimizer(grad(objective), self.init_var_params,
                               step_size=self.lr, num_iters=self.iters,
                               callback=callback)

        print('smallest MHD : {:.3f} for arch {}'.format(min(d.MHD_test_list),
                                                         arch_string(self.arch)))

if __name__ == '__main__':

    vars = ['PBLH']
    DL = DataLoader()
    max = np.mean

    x_train, y_train, x_test, y_test = DL.load_data(train=[[2016, 4]],
                                                    test=[[2016, 5]])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(max(x_train[:, 0]), max(x_train[:, 1]), max(x_train[:, 2]), max(x_train[:, 3]))

    vars = DL.train_vars
    arch = [len(vars),  21, 21, 1]
    save_dir = os.path.join(os.getcwd(), 'plots')

    model = BayesianNeuralNet(nn_arch=arch, save_dir=save_dir, labels=DL.labels)
    model.train(x_train, y_train, x_test, y_test)

