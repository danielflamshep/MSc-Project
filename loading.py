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


class DataLoader:
    def __init__(self, lat=44.75, lon=-80.3125, height=1, grad_height=1,
                 dates=[[2013, 5, 6, 7], [2014, 5, 6, 7]],
                 when='20132014050607', add_time=True, add_day=False,
                 scale='none', finite_diff='center', ignore=None):
        heights = [-0.006, 0.058, 0.189, 0.320, 0.454, 0.589, 0.726, 0.864,  # 16 heights
                   1.004, 1.146, 1.290, 1.436, 1.584, 1.759, 1.988, 2.249]
        self.heights=heights
        if ignore:
            self.ignore = ignore
        else:
            self.ignore = 'none'
        self.grad_height = grad_height
        self.scale_type = scale
        self.times = np.linspace(0.5, 23.5, 24)[1::3]
        self.dates = list_dates(dates)  # dates = [[2013,5,6,7],[2014,5,6,7]]
        self.ground_vars = ['TS', 'SWGDN']
        self.level_vars = ['T', 'U', 'V', 'QV', 'P', 'PT', 'UV']
        self.grad_vars = ['U', 'V', 'PT', 'UV']
        self.all_level_vars = [var+str(i) for var in self.level_vars for i in range(15)]
        self.all_grad_vars = [var+'G'+str(i) for var in self.grad_vars for i in range(1, 14)]
        self.height_level_vars = [var + str(height) for var in self.level_vars]
        self.height_grad_vars = [var + 'G' + str(grad_height) for var in self.grad_vars]
        self.vars = self.ground_vars + self.all_level_vars + self.all_grad_vars
        self.all_vars = self.vars + ['PBLH']+['TIME']

        hrs_idx = {'halfnight': [0, 2, 3, 4, 5, 6], 'allnight': [3, 4, 5, 6],
                   'halfday': [4, 5, 6, 7],         'midnight': [1, 2, 3, 4, 5, 6]}
        dir_load = os.path.join(os.getcwd(), 'data')
        fname = when + 'LAT' + str(lat) + 'LON' + str(lon) + 'LVL0-14vers2.npy'
        data = np.load(os.path.join(dir_load, fname), encoding='latin1')[()]

        g, gas_const, P_0, r = 9.81, 287.058, 1e3, 0.61
        # GET PRESSURES, POTENTIAL TEMPERATURE, WIND SPEED MAGNITUDE for heights 0 - 14
        for date in self.dates:
            data[date + 'P0'] = data[date + 'PS']
            data[date + 'UV0'] = data[date + 'U0'] ** 2 + data[date + 'V0'] ** 2
            data[date + 'PT0'] = data[date + 'T0'] * (P_0 / data[date + 'P0']) ** 0.286
            data[date + 'PTV0'] = (1-r* data[date+'QV0'])*data[date + 'PT0']

            for lvl in range(1, len(heights)-1):

                h = heights[lvl - 1] - heights[lvl]
                T_ave = 0.5 * (data[date + 'T' + str(lvl - 1)] + data[date + 'T' + str(lvl)])
                P_lower_level = data[date + 'P' + str(lvl - 1)]
                data[date + 'P' + str(lvl)] = P_lower_level * np.exp(g * h / gas_const / T_ave)

                U_level, V_level = data[date + 'U' + str(lvl)], data[date + 'V' + str(lvl)]
                data[date + 'UV' + str(lvl)] = U_level ** 2 + V_level ** 2

                T_level = data[date + 'T' + str(lvl)]
                P_level = data[date + 'P' + str(lvl)]
                QV_level = data[date + 'QV' + str(lvl)]
                PT = T_level * (P_0 / P_level) ** 0.286
                data[date + 'PT' + str(lvl)] = PT
                data[date + 'PTV' + str(lvl)] = (1-r*QV_level)*PT
        print('Calculated Pressures, potential temperatures and wind speed magnitudes')

        # GET VERTICAL GRADIENTS BY FINITE DIFFERENCES
        for date, lvl in product(self.dates, range(1, len(heights) - 2)):
            for var in ['U', 'V', 'PT', 'PTV']:
                if finite_diff == 'forward':
                    delta_var = data[date + var + str(lvl + 1)] - data[date + var + str(lvl)]
                    h = heights[lvl + 1] - heights[lvl]
                    data[date + var + 'G' + str(lvl)] = delta_var/h
                elif finite_diff == 'back':
                    delta_var = data[date + var + str(lvl)] - data[date + var + str(lvl - 1)]
                    h = heights[lvl] - heights[lvl - 1]
                    data[date + var + 'G' + str(lvl)] = delta_var/h
                else:
                    delta_var = data[date + var + str(lvl + 1)] - data[date + var + str(lvl - 1)]
                    h = heights[lvl + 1] - heights[lvl - 1]
                    data[date + var + 'G' + str(lvl)] = delta_var/h
            data[date + 'UVG' + str(lvl)] = data[date + 'UG' + str(lvl)] ** 2 + \
                                            data[date + 'VG' + str(lvl)] ** 2
        print('Calculated Vertical Gradients')

        # ADD TIME or DAY
        for date in self.dates:  # date is like '201305'
            days = int(data[date + 'TS'].shape[0] / 8)
            if add_time:
                data[date + 'TIME'] = np.tile(self.times, days)
            if add_day:
                data[date + 'DAY'] = np.array([[i] * 8 for i in range(1, days + 1)]).reshape(-1)


        # TRIM DATA
        for date in self.dates:
            data[date + 'shape'] = [data[date + var].shape[0] for var in self.vars]
            print(date, data[date + 'shape'])
            if max(data[date + 'shape']) != min(data[date + 'shape']):
                for var in (self.vars+['PBLH']):
                    data[date + var] = data[date + var][:min(data[date + 'shape'])]

        if ignore:
            for date in self.dates:  # date is like '201305'
                days = int(data[date + 'TS'].shape[0] / 8)
                n = hrs_idx[ignore]
                idx = [n[i] + 8 * k for k in range(days) for i in range(len(n))]
                for var in self.all_vars:
                    data[date + var] = data[date + var][idx]
                    print(date + var, data[date + var].shape[0])

        # SCALING
        self.scale = {'TS': 300.0, 'SWGDN': 1000.0, 'PBLH': 1000.0, 'TIME': 10.0,
                      'QV': 0.01, 'U': 10.0, 'V': 10.0, 'UV': 10.0,
                      'P': 900.0, 'T': 100.0, 'PT': 100.0,
                      'UG': 10.0, 'VG': 10.0, 'UVG': 100.0, 'PTG': 10.0}

        for date in self.dates:
            data[date + 'PBLH'] /= self.scale['PBLH']

        for date, var in product(self.dates, self.ground_vars):
            if scale == 'custom':
                data[date + var] /= self.scale[var]
            elif scale == 'maxmin':
                min_pblh = min(data[date + 'PBLH'])
                max_pblh = max(data[date + 'PBLH'])
                X = data[date + var]
                data[date + var] = MMS(X, feature_range=(min_pblh, max_pblh))
            elif scale == 'remove_mean':
                data[date + var] -= np.mean(data[date + var])

        for date, var, lvl in product(self.dates, self.level_vars, range(len(heights)-1)):
            if scale == 'custom':
                data[date + var + str(lvl)] /= self.scale[var]
            elif scale == 'maxmin':
                min_pblh = min(data[date + 'PBLH'])
                max_pblh = max(data[date + 'PBLH'])
                X = data[date + var + str(lvl)]
                data[date + var + str(lvl)] = MMS(X, feature_range=(min_pblh, max_pblh))
            elif scale == 'remove_mean':
                data[date + var + str(lvl)] -= np.mean(data[date + var + str(lvl)])

        for date, var, lvl in product(self.dates, self.grad_vars, range(1, len(heights) - 2)):
            if scale == 'custom':
                data[date + var + 'G' + str(lvl)] /= self.scale[var]
            elif scale == 'maxmin':
                min_pblh = min(data[date + 'PBLH'])
                max_pblh = max(data[date + 'PBLH'])
                X = data[date + var + 'G' + str(lvl)]
                data[date + var + 'G' + str(lvl)] = MMS(X, feature_range=(min_pblh, max_pblh))
            elif scale == 'remove_mean':
                data[date + var + str(lvl)] -= np.mean(data[date + var + str(lvl)])

        #print('scaled : '+var+' on '+date)

        self.data = data

    def check_pblh(self, Ric=0.3):
        data=self.data
        g = 9.81 # mixing ratio
        lvls = range(1, len(self.heights)-2)
        for date, lvl in product(self.dates, lvls):
            PTV = data[date+'PTV'+str(lvl)]
            PTVG = data[date+'PTVG'+str(lvl)]
            UVG = data[date+'UVG'+str(lvl)]
            RI = (g*PTVG)/(PTV*UVG)
            RIc = RI > Ric
            if date =='201406':
                print(date+str(lvl)+'PTV+PTVG+UVG',
                np.mean(PTV),np.mean(PTVG),np.mean(UVG))
            data[date+'RIc'+str(lvl)] = RIc

        for date in self.dates:
            list_Ric = [data[date+'RIc'+str(lvl)][:, None] for lvl in lvls]
            RIc = np.concatenate(list_Ric, axis=1).T  # shape LVL x HRS
            print(RIc.shape[0])
            for i, j in product(range(RIc.shape[0]), range(RIc.shape[1])):
                if RIc[i, j]:
                    print(date+' level '+str(i)+' day = {} hour= {}'.format(j/8, j%8 ))
                    RIc[i, j] = self.heights[i]
            PBLH_check = np.min(RIc, axis=0)
            if date =='201406':
                print(PBLH_check)
            data[date + 'PBLHc'] = PBLH_check
        self.data=data

    def load_data(self, train, test, input_vars=None, interpolation=0, plot=False, plot_dir=None):
        # train and test of form train = [[2013,5,6,7],[2014,5,6,7]]

        train_dates = list_dates(train)
        test_dates = list_dates(test)
        dates = train_dates + test_dates

        if input_vars is None:
            input_vars = self.ground_vars + self.height_grad_vars + \
                         self.height_level_vars
        else:
            input_vars = input_vars

        if interpolation == 0:
            pass
        else:
            if plot:
                if plot_dir:
                    save_dir = plot_dir
                else:
                    plot_dir = os.path.join(os.getcwd(), 'plots', 'interpolation')
                    if not os.path.exists(plot_dir):
                        os.mkdir(plot_dir)

                    folder = 'interpolation_' + str(interpolation) + \
                             'train_dates_' + to_string(train) + \
                             'test_dates_' + to_string(test)

                    save_dir = os.path.join(plot_dir, folder)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

            for date, var in product(dates, input_vars + ['PBLH']):

                print('fitting interpolation for ' + date + var)
                X = np.arange(self.data[date + var].shape[0])[:, None]
                y = self.data[date + var]

                gp = GPR(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                         n_restarts_optimizer=9, random_state=1337)
                gp.fit(X, y)

                x = np.linspace(0, X.shape[0], X.shape[0] * interpolation)[:, None]
                y_pred, sigma = gp.predict(x, return_std=True)
                self.data[date + var] = y_pred

                if plot:
                    plt.clf()
                    plt.plot(X, y, 'r-', lw=2, label=u'real ' + var)
                    plt.plot(x, y_pred, 'b-', lw=1, label=u'interpreted ' + var)
                    plt.plot(X, y, 'ro', label=u'real ' + var)
                    plt.plot(x, y_pred, 'bx', label=u'interpreted ' + var)
                    plt.fill(np.concatenate([x, x[::-1]]),
                             np.concatenate([y_pred - 1.9600 * sigma,
                                             (y_pred + 1.9600 * sigma)[::-1]]),
                             alpha=.5, fc='b', ec='None', label='95% CI')
                    plt.xlabel(date)
                    plt.ylabel(var)
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(save_dir, date + var + '.jpg'))
                    print('plotted : ' + date + var)

        for var in input_vars:
            train_list = [self.data[date + var] for date in train_dates]
            test_list = [self.data[date + var] for date in test_dates]
            self.data['train' + var] = np.concatenate(train_list)
            self.data['test' + var] = np.concatenate(test_list)

        train_inputs_list = [self.data['train' + var][:, None] for var in input_vars]
        test_inputs_list = [self.data['test' + var][:, None] for var in input_vars]

        y_train_list = [self.data[date + 'PBLH'][:, None] for date in train_dates]
        y_test_list = [self.data[date + 'PBLH'][:, None] for date in test_dates]

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
    scale = 'none'
    ignore = 'none'
    vars = ['RIc' +str(lvl) for lvl in range(1,14)]
    DL = DataLoader(scale=scale, ignore=None)
    DL.check_pblh()
    DL.plot_time_series(dates=[[2014, 6]], plot_vars=vars, num=1)
    # DL.load_data(train=[[2014,5,6]], test=[[2014,7]], interpolation=5)
