import os
from neural_net import NeuralNet
from loading import DataLoader
from mlp import list_archs
from util import to_string_nn

train_mns = [[2014, 6]]
test_mns = [[2014, 7]]
run = 3
root = os.path.join(os.getcwd(), 'plots')
for scale in ['maxmin', 'remove_mean']:
    for ignore in [None, 'halfnight', 'halfday', 'allnight', 'midnight']:
        dl = DataLoader(scale=scale, ignore=ignore)
        all_vars = dl.all_vars
        run_dir = os.path.join(root, 'run_' + str(run))
        if ignore is None:
            ignore = 'none'
        scale_dir = os.path.join(run_dir, 'scale_' + scale + '_ignore_'+ignore)
        ts_dir = os.path.join(scale_dir, 'TS')
        for dir in [run_dir, scale_dir, ts_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

        dl.plot_time_series(dates=[[2014, 6, 7]], plot_dir=ts_dir)

        for arch in [[1,25,1],[1,30,30,1],[1,80,80,1]]:
            arch_dir = os.path.join(scale_dir, 'arch_' + to_string_nn(arch))
            if not os.path.exists(arch_dir):
                os.mkdir(arch_dir)

            for var in all_vars:
                x_train, y_train, x_test, y_test = dl.load_data(train_mns, test_mns, input_vars=[var])
                model = NeuralNet(nn_arch=arch, input_var=var, save_dir=arch_dir)
                model.train(x_train, y_train, x_test, y_test)