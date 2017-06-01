import tensorflow as tf

def linear(input_, output_size, in_dim, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  #shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    # matrix = tf.get_variable("Matrix", [in_dim, output_size], tf.float32,
    #              tf.random_normal_initializer(stddev=stddev))
    matrix = tf.get_variable("Matrix", [in_dim, output_size], tf.float32,
                             tf.contrib.layers.xavier_initializer(uniform=False))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, [matrix, bias]
    else:
      return tf.matmul(input_, matrix) + bias


def deconv2d(input_, output_shape, out_channels,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        #w = tf.get_variable('w', [k_h, k_w, out_channels, input_.get_shape()[-1]],
        #                    initializer=tf.random_normal_initializer(stddev=stddev))
        w = tf.get_variable('w', [k_h, k_w, out_channels, input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        print deconv.shape
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, [w, biases]
        else:
            return deconv

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon  = epsilon
          self.momentum = momentum
          self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train, scope=self.name)