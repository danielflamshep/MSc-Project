import numpy as np


def compare_dates(pblh_dates, inputs_dates, train, test):

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


def compare_times(date, pblh, inputs, close=0.5):

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


def match_up_times(pblh_data, inputs_data, vars, train, test):

    data = {}
    for date in train + test:
        key = date+'HR'
        idx_pblh, idx_inputs = compare_times(date, pblh_data[key], inputs_data[key])
        data[date+'PBLH'] = pblh_data[date+'PBLH'][idx_pblh]
        for var in vars:
            data[date+var] = inputs_data[date+var][idx_inputs]

    return data


# y = np.load('pblh.npy')[()]
# x = np.load('inputs.npy')[()]
# date = '20130612'
# pblh_test = y[date+'TIME']
# inputs_test = x[date+'HR']
#
# compare_times(date, pblh_test, inputs_test)
