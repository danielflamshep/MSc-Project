import numpy as np
import os


def list_dates(dates):
    """ [[2013, 5, 6, 7], [2014, 5, 6, 7]] --> ['201305','201306',...] """
    def f(x, y):
        if y > 9:
            return str(x) + str(y)
        else:
            return str(x) + '0' + str(y)
    return [f(x[0], x[i]) for x in dates for i in range(1, len(x))]


def to_string(t):
    """ [[2013,5,6,7],[2014,5,6,7]] ->'20130506072014050607' """
    return ''.join(str(x[0])+'0'+'0'.join(str(i) for i in x[1:]) for x in t)


def to_date_mnday(num_day, year):
    if year % 4 == 0:
        days  = [31, 29, 31, 30,  31,  30,  31,  31,  30,  31,  30,  31]
        sdays = [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        days  = [31, 28, 31, 30,  31,  30,  31,  31,  30,  31,  30,  31]
        sdays = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    for i in range(len(days)):
        if num_day < sdays[i]:
            #print(i, sdays[i - 1])
            if num_day < 31:
                day = num_day
            else:
                day = num_day - sdays[i-1]
            month = i+1
            break
        elif num_day == sdays[i]:
            day = days[i]
            month = i+1
            break

    return string_with_zero(month)+string_with_zero(day)

def to_date(x):
    # [[2013, 6, 7],[2014,5,6,7]] -> 201306072014050607
    when = ''
    for d in x:
        when += str(d[0])+''.join('0'+str(d[i]) for i in range(1, len(d)))
    return when

def string_with_zero(y):
    if y > 9:
        return str(y)
    else:
        return '0' + str(y)

def get_fname(LAT, LON, vers):
    return 'LAT' + str(LAT) + \
           'LON' + str(LON) + \
           'LVL0-14vers'+ str(vers)


def fname_nn(train, test, arch):
    return 'train_' + to_date(train) + \
           '_test_' + to_date(test) + \
           '_NN_' + '.'.join(str(x) for x in arch[1:-1])


def date_string(t):
    """ [[2013,7],[2014,7]] ->'2013 JULY 2014 JULY' """

    month = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR',
             5: 'MAY', 6: 'JUNE', 7: 'JULY', 8: 'AUG',
             9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

    return ''.join(str(x[0])+' '+', '.join(month[i] for i in x[1:]) for x in t)


def arch_string(arch):
    return 'BNN-'+'.'.join(str(b) for b in arch[1:-1])


class Dataholder(object):
    def __init__(self, y_train, y_test,
                 loss, save_dir, arch):
        self.arch = arch
        self.loss = loss
        self.iters = 0
        self.p_test = 0
        self.p_train = 0
        self.loss_test = 0
        self.loss_train = 0
        self.MHD_test = 0
        self.MHD_train = 0
        self.labels = None
        self.y_test = y_test
        self.y_train = y_train
        self.loss_test_list = []
        self.loss_train_list = []
        self.MHD_train_list = []
        self.MHD_test_list = []
        self.save_dir = save_dir

    def labels(self, label_dict):
        self.labels = label_dict

    def get_ptest(self, value):
        self.p_test = value
        loss = self.loss(value, self.y_test)
        mhd = np.mean(np.abs(self.y_test - value))
        self.loss_test_list.append(loss)
        self.loss_test = loss
        self.MHD_test = mhd
        self.MHD_test_list.append(mhd)

    def get_ptrain(self, value):
        self.p_train = value
        loss = self.loss(value, self.y_train)
        mhd = np.mean(np.abs(self.y_train - value))
        self.loss_train_list.append(loss)
        self.loss_train = loss
        self.MHD_train = mhd
        self.MHD_train_list.append(mhd)

    def get_fname(self):
        fname = 'MHD={:.3f} '.format(self.MHD_test) +\
                arch_string(self.arch) +\
                ' iters{}'.format(self.iters) +\
                '.jpg'
        return fname

    def mk_dir(self):
        folder = self.labels['train'][11:]+"--"+self.labels['test'][11:]
        fpath=os.path.join(self.save_dir, folder)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        return os.path.join(fpath, self.get_fname())
