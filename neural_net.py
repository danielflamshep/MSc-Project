from __future__ import absolute_import
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from optimizers import adam
import plotting
from lidar_data_loader import DataLoader
from util import arch_string, Dataholder


class NeuralNet:
    def __init__(self, nn_arch, input_var, save_dir,
                 lr=0.005, iterations=1000, scale=1.0, seed=15,
                 activation=np.tanh,  plot=plotting.plot_pblh,
                 labels=None):
        self.labels = labels
        self.arch = nn_arch
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

    @staticmethod
    def forward(params, inputs, f=np.tanh):
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = f(outputs)
        return outputs

    def loss(self, p, y):
        return 0.5 * np.sum((y - p) ** 2)/y.shape[0]

    def train(self, x_train, y_train, x_test, y_test,
              optimizer=adam, plot_during=True):

        nn = self

        d = Dataholder(y_train, y_test, nn.loss, nn.save_dir, nn.arch)
        d.labels = nn.labels

        def objective(params, t):
            p = nn.forward(params, x_train)
            return nn.loss(p, y_train)

        fig, axes = plotting.set_up()

        def callback(params, t, g):
            d.get_ptrain(nn.forward(params, x_train))
            d.get_ptest(nn.forward(params, x_test))

            if plot_during:
                nn.plot(d, axes)

            if t % 100.0 == 0:
                print("{}| TEST MHD {:.3f}".format(t, d.MHD_test))

            if d.MHD_test < 0.135:
                d.iters = t
                nn.plot(d, axes, draw=False, save=True)
                print('plotted with MHD: {:.3f}'.format(d.MHD_test))

        self.params = optimizer(grad(objective),
                                self.params, step_size=self.lr,
                                num_iters=nn.iters, callback=callback)

        return self.params

if __name__ == '__main__':
    train_mns = [[2014, 6]]
    test_mns = [[2014, 7]]
    scale = 'maxmin'
    h = 5
    ignore = 'allnight'
    intrp = 100
    dl = DataLoader(scale=scale, ignore=ignore, grad_height=h, height=h)
    input_vars = dl.height_grad_vars+dl.ground_vars+['TIME']
    if ignore is None: ignore = 'none'
    arch = [len(input_vars), 100, 100, 1]

    arch_dir = os.path.join(os.getcwd(), 'plots', 'new')\
               +r'\\scale_'+scale +'_ignore'+ignore + \
               '_intrp_'+str(intrp)+'_nn_'+arch_string(arch)

    x_train, y_train, x_test, y_test = dl.load_data(train_mns, test_mns,
                                                    input_vars=input_vars,
                                                    interpolation=intrp)
    model = NeuralNet(nn_arch=arch, input_var=input_vars, save_dir=arch_dir)
    model.train(x_train, y_train, x_test, y_test)


