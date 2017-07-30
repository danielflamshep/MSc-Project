import numpy as np


def compare_day(pblh, inputs):

    if pblh.shape[0] >= inputs.shape[0]:
        d = pblh[:, None] - inputs.T[None, :]
        idx_inputs = np.argmin(np.abs(d), axis=1)
        idx_pblh = np.arange(pblh.shape[0])

    else:
        d = inputs[:, None] - pblh.T[None, :]
        idx_inputs = np.argmin(np.abs(d), axis=1)
        idx_pblh = np.arange(inputs.shape[0])

    print(idx_pblh, idx_inputs)

    return idx_pblh, idx_inputs




compare_day(a, b)