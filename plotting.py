import matplotlib.pyplot as plt
import os
import numpy as np
from numpy import arange as ar

def ave_pblh(y,p): return np.mean(np.abs(y-p))
def pblh_diff(y,p): return np.abs(y-p)


def to_date(x):
    when = ''
    for d in x:
        when += ' YR: ' + str(d[0])+' MN ' + ''.join('0'+str(d[i]) for i in range(1, len(d)))
    return when


def set_up(show=True):
    fig = plt.figure(figsize=(20, 8), facecolor='white')
    ax = fig.add_subplot(211, frameon=True)
    bx = fig.add_subplot(212, frameon=True)

    if show:
        plt.show(block=False)

    axes = (ax, bx)

    return fig, axes


def plot_pblh(d, axes, draw=False, save=False):

    labels = d.labels

    ax, bx = axes

    ax.cla()
    bx.cla()

    train_labels = labels['train_labels']
    test_labels = labels['test_labels']
    x_train = labels['train_ticks']
    x_test = labels['test_ticks']

    labels_dict = {'fontname': 'courier new', 'weight': 'bold', 'size': 25}

    # PLOT TRAINING TIME SERIES
    ts_train = ar(len(d.y_train))[:, None]

    ax.plot(ts_train, d.y_train, 'b.', ms=12)
    ax.plot(ts_train, d.p_train, 'g.', ms=12, label='NN PBL heights')

    # ax.plot(ts_train, y_train, color='b', lw=1)
    # ax.plot(ts_train, p_train, color='g', lw=1)

    # ax.set_ylim([0, 2])
    ax.set_ylabel('PBL heights (km)', labels_dict)
    ax.set_xlabel('Days in Training Month(s):'+labels['train'][11:], labels_dict)
    ax.set_title(labels['train'], labels_dict)

    # PLOT TESTING TIME SERIES

    ts_test = ar(len(d.y_test))[:, None]
    bx.plot(ts_test, d.y_test,  'b.', ms=12,  label='MiniMPL PBL heights')
    bx.plot(ts_test, d.p_test,  'g.', ms=12)

    # bx.plot(ts_test, y_test, color='b', lw=1)
    # bx.plot(ts_test, p_test, color='g', lw=1)

    # bx.set_ylim([0, 2])
    bx.set_ylabel('PBL heights (km)', labels_dict)
    bx.set_xlabel('Days in Testing Month(s):'+labels['test'][10:], labels_dict)
    bx.set_title(labels['test'], labels_dict)

    ticks_dict = labels_dict
    ticks_dict['size'] = 18

    ax.set_xticks(x_train)
    ax.set_xticklabels(train_labels, ticks_dict)

    bx.set_xticks(x_test)
    bx.set_xticklabels(test_labels, ticks_dict)

    ax.grid('off', axis='y')
    bx.grid('off', axis='y')

    anchor = (0., 1., 1., .0)

    bx.legend(bbox_to_anchor=anchor, loc=3, ncol=2, frameon=True,
              prop={'family': 'courier new', 'weight': 'bold', 'size': 20})

    ax.legend(bbox_to_anchor=(1., -0.15), ncol=2, frameon=True,
              prop={'family': 'courier new', 'weight': 'bold', 'size': 20})

    plt.tight_layout()
    ax.tick_params(axis='y', labelsize=15)
    bx.tick_params(axis='y', labelsize=15)

    if draw:
        plt.draw()
        plt.pause(1.0 / 60.0)

    if save:
        path = d.mk_dir()
        plt.savefig(path, bbox_inches='tight')


def plot_pblh_diff(y_train, y_test, p_train, p_test, t, ax, bx):

    ax.cla()
    bx.cla()

    # PLOT TRAINING TIME SERIES
    ts_train = ar(len(y_train))  # TIME SERIES INPUTS
    ax.plot(ts_train, pblh_diff(y_train, p_train), 'b', lw=2)
    ax.set_ylim([0, 2])
    ax.set_ylabel('$|y_{PBLH}-p_{PBLH}|$ in km')
    ax.set_title('training data  ')

    # PLOT TESTING TIME SERIES
    ts_test = ar(len(y_test))
    bx.plot(ts_test, pblh_diff(y_test, p_test), 'g', lw=2)
    bx.set_ylim([0, 2])
    bx.set_ylabel('|y_{PBLH}-p_{PBLH}|$ in km')
    bx.set_title('test data : ')

    plt.pause(1.0 / 60.0)


def plot_ave_pblh_diff(y_train, y_test, ax, bx, p_train, p_test, t):

    pblh_train = ave_pblh(y_train, p_train)
    pblh_test = ave_pblh(y_test, p_test)

    ax.plot([t], [pblh_train], marker='o', ms=2, c='g')
    ax.set_ylim([0.2, 0.5])
    ax.set_ylabel('PBLH diff in km')
    ax.set_title('training data : JUNE 2013')
    plt.draw()
    # PLOT TESTING TIME SERIES
    bx.plot(t, pblh_test, marker='o', ms=2, c='r')
    bx.set_ylim([0.2, 0.5])
    bx.set_ylabel('PBLH diff in km')
    bx.set_title('test data : JULY 2013')

    plt.pause(1.0 / 60.0)


def plot_mse(mse, file):
    plt.clf()
    plt.plot(ar(len(mse)), mse, 'r')
    plt.savefig(file+'mse.jpg')


def plot_interpolation(X, x, y, y_pred, sigma):
    fig = plt.figure()
    plt.plot(X, y, 'r-', lw=2, label=u'real pblh')
    plt.plot(x, y_pred, 'b-', lw=1, label=u'interpreted pblh')
    plt.plot(X, y, 'ro', label=u'real pblh')
    plt.plot(x, y_pred, 'bx', label=u'interpreted pblh')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('time')
    plt.ylabel('PBLH')
    plt.legend(loc='upper left')
    plt.show()