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


class NeuralNet:
    def __init__(self, nn_arch, input_var, save_dir,
                 lr=0.005, iterations=1000, scale=1.0, seed=15,
                 activation=np.tanh,  plot=plotting.plot_pblh):
        self.nn_arch = nn_arch
        self.activation = activation
        self.input_var = '+'.join(a for a in input_var)
        self.save_dir = save_dir
        self.lr = lr
        self.iterations = iterations
        self.plot = plot
        self.Trained = False
        rs = npr.RandomState(seed)
        self.params = [(rs.randn(i, j) * scale, rs.randn(j) * scale)  # weight bias tuple
                       for i, j in zip(nn_arch[:-1], nn_arch[1:])]

    def train(self, x_train, y_train, x_test, y_test,
              optimizer=adam, loss=mse, plot_during=True):

        activation, plot, iters = self.activation, self.plot, self.iterations
        input_var, save_dir = self.input_var, self.save_dir

        def objective(params, t):
            p = forward(params, x_train, activation)
            return loss(p, y_train)

        mse_test_list, pblh_train_diff = [], []

        fig, ax, bx = plotting.set_up()

        def callback(params, t, g):

            p_train = forward(params, x_train, activation)
            p_test = forward(params, x_test, activation)
            mse_train = loss(p_train, y_train)
            mse_test = loss(p_test, y_test)
            mse_test_list.append(mse_test)

            plot(y_train, y_test, p_train, p_test, t, ax, bx)

            if t % 100.0 == 0:
                print("ITER {} TRAIN MSE {:.5f} TEST MSE {:.5f}".format(t, mse_train, mse_test))

            if t == iters-1:
                metric = np.mean(np.abs(y_test - p_test))
                plot(y_train, y_test, p_train, p_test, t, ax, bx)
                fname = input_var + 'avePBLHdiff=' + str(metric)+'.jpg'
                plt.savefig(save_dir+ fname)
                print('mean PBLH diff for '+input_var, metric)

        self.params = optimizer(grad(objective), self.params, step_size=self.lr,
                                num_iters=iters, callback=callback)

        return

# if __name__ == '__main__':
#     train_mns = [[2014, 6]]
#     test_mns = [[2014, 7]]
#     scale = 'maxmin'
#     h = 5
#     ignore = 'allnight'
#     intrp = 100
#     dl = DataLoader(scale=scale, ignore=ignore, grad_height=h, height=h)
#     input_vars = dl.height_grad_vars+dl.ground_vars+['TIME']
#     if ignore is None: ignore = 'none'
#     arch = [len(input_vars), 100, 100, 1]
#
#     arch_dir = os.path.join(os.getcwd(), 'plots', 'new')+r'\\scale_'+scale +\
#                                                         '_ignore'+ignore + '_intrp_'+str(intrp)+\
#                                                         '_nn_'+to_string_nn(arch)
#
#     x_train, y_train, x_test, y_test = dl.load_data(train_mns, test_mns, input_vars=input_vars, interpolation=intrp)
#     model = NeuralNet(nn_arch=arch, input_var=input_vars, save_dir=arch_dir)
#     model.train(x_train, y_train, x_test, y_test)


