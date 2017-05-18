import tensorflow as tf


def model(x):

    # Input Tensor Shape: [batch_size, 225, 202, 72, 1]
    # Output Tensor Shape: [batch_size, 225, 202, 72, 5]
    convolution_layer_1 = tf.layers.conv3d(
                                        inputs=x,
                                        filters=5,
                                        kernel_size=[5, 5, 5],
                                        strides=(1, 1, 1),
                                        padding='valid',
                                        data_format='channels_last',
                                        dilation_rate=(1, 1, 1),
                                        activation=tf.nn.tanh()
                                    )

    # Input Tensor Shape: [batch_size, 225, 202, 72, 5]
    # Output Tensor Shape: [batch_size, 225, 202, 5]
    out_shape = convolution_layer_1.get_shape().as_list()
    input_layer_2 = tf.reshape(convolution_layer_1, out_shape[:3] + out_shape[-1])
    # or
    # input_layer_2 = tf.reduce_mean(convolution_layer_1, axis=3)

    # Input Tensor Shape: [batch_size, 225, 202, 5]
    # Output Tensor Shape: [batch_size, 225, 202, 15]
    convolution_layer_2 = tf.layers.conv2d(
                                        inputs=input_layer_2,
                                        filters=15,
                                        kernel_size=[5, 5],
                                        padding="same",
                                        activation=tf.nn.tanh())

    # Input Tensor Shape: [batch_size, 225, 202, 15]
    # Output Tensor Shape: [batch_size, 225, 202]
    outputs = tf.reduce_mean(convolution_layer_2, axis=-1)

    return outputs



