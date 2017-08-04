import matplotlib.pyplot as plt
import numpy as np
from numpy import arange as ar

def ave_pblh(y,p): return np.mean(np.abs(y-p))
def pblh_diff(y,p): return np.abs(y-p)


def to_date(x):
    when=''
    for d in x:
        when += ' YR: ' + str(d[0])+' MN ' + ''.join('0'+str(d[i]) for i in range(1, len(d)))
    return when


def set_up(show=True):
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    ax = fig.add_subplot(211, frameon=True)
    bx = fig.add_subplot(212, frameon=True)
    if show:
        plt.show(block=False)
    return fig, ax, bx


def plot_pblh(y_train, y_test,  p_train, p_test, t, ax, bx):

    ax.cla()
    bx.cla()

    # PLOT TRAINING TIME SERIES
    ts_train = ar(len(y_train))[:, None]
    #print(ts_train.shape)# TIME SERIES INPUTS
    ax.plot(ts_train, y_train, color='b', marker='.')
    ax.plot(ts_train, p_train, color='g', marker='.')
    ax.plot(ts_train, y_train, color='b', lw=1)
    ax.plot(ts_train, p_train, color='g', lw=1)
    ax.set_ylim([-1, 3])
    ax.set_ylabel('PBLH in km')
    ax.set_title('training data : ')

    # PLOT TESTING TIME SERIES
    ts_test = ar(len(y_test))[:, None]
    bx.plot(ts_test, y_test, color='b', marker='.')
    bx.plot(ts_test, p_test, color='g', marker='.')
    bx.plot(ts_test, y_test, color='b', lw=1)
    bx.plot(ts_test, p_test, color='g', lw=1)
    bx.set_ylim([-1, 3])
    bx.set_ylabel('PBLH in km')
    bx.set_title('test data : ')
    plt.draw()
    plt.pause(1.0 / 60.0)



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