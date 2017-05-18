import tensorflow as tf
import numpy as np
from core_model import model

data_dir = 'E:/physics Msc/summer research/code/data/'

lat, lon, vert = 250, 202, 72
epochs, batch_size, lr = 20, 5, 0.001

x = tf.placeholder(tf.float32, [None, lat, lon, vert, 1])
y = tf.placeholder(tf.float32, [None, lat, lon])
preds = model(x)
mean_PBLH_diff = tf.reduce_mean(tf.abs(y-preds))
loss = tf.losses.mean_squared_error(labels=y, predictions=preds)
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hist_test, hist_train = [], []

    inputs_train = np.load('../train_T.npz')
    target_train = np.load('../train_PBLH.npz')
    inputs_test = np.load('../test_T.npz')
    target_test = np.load('../test_PBLH.npz')

    rnd_idx = np.arange(inputs_train.shape[0])
    num_train_cases = inputs_train.shape[0]

    num_steps = int(np.ceil(num_train_cases / batch_size))
    for epoch in range(epochs):

        np.random.shuffle(rnd_idx)
        inputs_train = inputs_train[rnd_idx]
        target_train = target_train[rnd_idx]
        
        for step in range(num_steps):
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            batch_x = inputs_train[start: end]
            batch_y = target_train[start: end]
            sess.run([model, loss, optimizer], feed_dict={x: batch_x, y: batch_y})

        testMSE = loss.eval({x: inputs_test, y: target_test})
        trainMSE = loss.eval({x: inputs_train, y: target_train})

        hist_test.append(testMSE)
        hist_train.append(trainMSE)

        print('Epoch %01d | Train MSE = %.5f | Test MSE = %.5f' % (epoch+1, trainMSE, testMSE))

    print("Training Finished with Average Test PBLH diff:", mean_PBLH_diff.eval({x: inputs_test, y: target_test}))
