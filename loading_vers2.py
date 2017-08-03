import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange as ar
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import minmax_scale as MMS
from util import list_dates, to_string
import seaborn as sns
from neural_net import NeuralNet


class DataLoader:
    def __init__(self, vars=None):
        if vars is None:
            vars = ['P', 'QV', 'T', 'SI', 'U', 'HR']
        self.train_vars = vars
        self.pblh_data = np.load(os.path.join(os.getcwd(),'pblh.npy'))[()]
        self.inputs_data = np.load(os.path.join(os.getcwd(), 'inputs.npy'))[()]
        self.pblh_dates = self.pblh_data['dates']
        self.inputs_dates = self.inputs_data['dates']
        self.scale = {'P': 1e3, 'QV': 1e2, 'T': 1e2, 'SI': 1e2, 'U': 1e2, 'HR': 1e1}

    def compare_dates(self, train, test):

        pblh_dates, inputs_dates = self.pblh_dates, self.inputs_dates

        real_train = []
        real_test = []

        print('training on :')
        for date in train:
            for date_with_day in pblh_dates:
                if date_with_day in inputs_dates and date in date_with_day:
                    real_train.append(date_with_day)
                    print(date_with_day)

        print('testing on :')
        for date in test:
            for date_with_day in pblh_dates:
                if date_with_day in inputs_dates and date in date_with_day:
                    real_test.append(date_with_day)
                    print(date_with_day)

        return real_train, real_test

    @staticmethod
    def _compare_times(date, pblh, inputs, close=0.5):

        idx_pblh = []
        idx_inputs = []

        d = pblh[:, None] - inputs.T[None, :]  # get time difference
        d = np.abs(d)  # only positive
        p = d < close  # which are less than half hr apart
        min_idx = np.argsort(d)  # idx of the closest times

        for i, idx_i in enumerate(min_idx):
            for j in idx_i:
                if p[i, j] and j not in idx_inputs:
                    idx_inputs.append(j)
                    idx_pblh.append(i)
                    break

        print('For {} there are {} matches at times:'.format(date, len(idx_pblh)))
        for i, j in zip(idx_pblh, idx_inputs):
            print(pblh[i], inputs[j])

        return idx_pblh, idx_inputs

    def _match_up_times(self, train, test):

        pblh_data, inputs_data, vars = self.pblh_data, self.inputs_data, self.train_vars
        scale = self.scale
        data = {}
        for date in train + test:
            key = date + 'HR'
            idx_pblh, idx_inputs = self._compare_times(date, pblh_data[key], inputs_data[key])
            data[date + 'PBLH'] = pblh_data[date + 'PBLH'][idx_pblh]

            for var in vars:
                data[date + var] = inputs_data[date + var][idx_inputs]/scale[var]

        return data

    def load_data(self, train, test, input_vars=None):
        # train and test of form train = [[2013,5,6,7],[2014,5,6,7]]

        vars = self.train_vars
        train_dates = list_dates(train)
        test_dates = list_dates(test)

        train_dates, test_dates = self.compare_dates(train_dates, test_dates)

        data = self._match_up_times(train_dates, test_dates)

        if input_vars is None:
            input_vars = self.train_vars
        else:
            input_vars = input_vars

        for var in input_vars:
            train_list = [data[date + var] for date in train_dates]
            test_list = [data[date + var] for date in test_dates]
            data['train' + var] = np.concatenate(train_list)
            data['test' + var] = np.concatenate(test_list)

        train_inputs_list = [data['train' + var][:, None] for var in input_vars]
        test_inputs_list = [data['test' + var][:, None] for var in input_vars]

        y_train_list = [data[date + 'PBLH'][:, None] for date in train_dates]
        y_test_list = [data[date + 'PBLH'][:, None] for date in test_dates]

        x_train = np.concatenate(train_inputs_list, axis=1)
        y_train = np.concatenate(y_train_list)

        x_test = np.concatenate(test_inputs_list, axis=1)
        y_test = np.concatenate(y_test_list)

        return x_train, y_train, x_test, y_test

    def plot_time_series(self, dates, plot_vars=None, plot_dir=None,
                            plot_pblh=False, num=0):

        plot_vars = self.vars if plot_vars is None else plot_vars

        if plot_dir:
            save_dir = plot_dir
        else:
            plot_dir = os.path.join(os.getcwd(), 'plots', 'timeseries')
            folder = self.scale_type + '_scale_' + str(num)
            save_dir = os.path.join(plot_dir, folder)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        for date, var in product(list_dates(dates), plot_vars):
            plt.clf()

            time = np.arange(self.data[date + var].shape[0])
            time_series = self.data[date + var]
            plt.scatter(time, time_series, c='b', marker='x')
            plt.plot(time, time_series, 'b', lw=1, label=var)
            print('plotted : ' + date + var + ' with ')

            if plot_pblh:
                pblh_time = np.arange(self.data[date + 'PBLH'].shape[0])
                pblh_time_series = self.data[date + 'PBLH']
                plt.plot(pblh_time, pblh_time_series, 'r', lw=1, label='PBLH')
                plt.scatter(pblh_time, pblh_time_series, c='r', marker='x')
                pearson_corr = np.corrcoef(time_series, pblh_time_series)[0, 1]
                plt.suptitle('pear_corr : ' + str(pearson_corr))
                print('plotted : ' + date + var + ' with ')
                print('# of PBLH < 100 m = ', sum(pblh_time_series < 0.100))
            plt.xlabel(date)
            plt.ylabel(var)
            plt.legend()
            plt.title('Time Series on ' + date + ' for ' + var)
            plt.savefig(os.path.join(save_dir, self.ignore+'TS' + date + var + '.jpg'))


if __name__ == '__main__':
    vars = ['PBLH']
    DL = DataLoader()
    max=np.mean
    # DL.plot_time_series(dates=[[2014, 6]], plot_vars=vars, num=1)
    x_train, y_train, x_test, y_test = DL.load_data(train=[[2014, 9]], test=[[2014, 10]])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(max(x_train[:,0]), max(x_train[:,1]) , max(x_train[:,2]), max(x_train[:,3]))
    vars = DL.train_vars
    arch = [len(vars), 6, 1]
    save_dir = os.path.join(os.getcwd(), 'plots')
    model = NeuralNet(nn_arch=arch, input_var=vars, save_dir=save_dir)
    model.train(x_train, y_train, x_test, y_test)