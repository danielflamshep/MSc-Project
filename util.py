import numpy as np

def list_dates(dates):
    ''' [[2013, 5, 6, 7], [2014, 5, 6, 7]] --> ['201305','201306',...]da'''
    def f(x, y):
        if y > 9:
            return str(x) + str(y)
        else:
            return str(x) + '0' + str(y)
    return [f(x[0], x[i]) for x in dates for i in range(1, len(x))]


def to_string(t):
    ''' [[2013,5,6,7],[2014,5,6,7]] ->'20130506072014050607' '''
    return ''.join(str(x[0])+'0'+'0'.join(str(i) for i in x[1:]) for x in t)


def to_date(x):
    # [[2013, 6, 7],[2014,5,6,7]] -> 201306072014050607
    when=''
    for d in x:
        when += str(d[0])+''.join('0'+str(d[i]) for i in range(1, len(d)))
    return when


def get_fname(LAT,LON,vers):
    return 'LAT' + str(LAT) + 'LON' + str(LON) + 'LVL0-14vers'+ str(vers)


def fname_nn(train, test, arch):
    return 'train_' + to_date(train) + '_test_' + to_date(test) + \
           '_NN_' + '.'.join(str(x) for x in arch[1:-1])


def to_string_nn(arch):
    return '_NN_'+'.'.join(str(b) for b in arch[1:-1])


