import autograd.numpy as np


def forward(params, inputs, f=np.tanh):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = f(outputs)
    return outputs


def mean(x): return sum(x)/len(x)


def mse(p, y): return 0.5 * np.sum((y - p) ** 2)/y.shape[0]


def early_stopping(iterations, loss, check_at=100, check=10):
    if (iterations + 1) % check_at == 0:
        #print('checking for early stopping with at loss : ',loss[-1])
        cond_a = loss[-1] > mean(loss[iterations-check:-1])
        cond_b = all(loss[-1] > s for s in loss[iterations-check:-1])
        #print(cond_a, cond_b)
        return any([cond_a, cond_b])
    else:
        return False


def gen_arch(l=7, neurons=[30,50,75,100,150,250,500]):
    archs=[]
    for neuron in neurons:
        archs +=[[6]+[neuron for i in range(1,j)]+[1] for j in range(2, l)]
    return archs


def gen_trains():
    trains=[[[2013]+[4+i for i in range(1,j)]]for j in [2,3,4]]+\
           [[[2014] + [4 + i for i in range(1, j)]] for j in [2, 3]]+ \
           [[[2013] + [4 + i for i in range(1, j)]] + [[2014] + [4 + i for i in range(1, j)]] for j in [2, 3]]+\
            [[[2013,7],[2014, 5, 6]],
             [[2013, 6, 7],[2014, 5, 6]],
             [[2013, 5, 6, 7],[2014, 5, 6]]]
    return trains


def list_archs():
    return [[1] + [5 * b] + [1] for b in [1, 2, 5, 10, 20]] +\
           [[1] + [5 * b, 5 * b] + [1] for b in [1, 2, 5, 10, 20]]

