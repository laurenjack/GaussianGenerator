import math
import numpy as np
from ops import *


class Generator:

    def __init__(self, conf):
        self.d = conf.d
        self.im_height = conf.ds_params.im_height
        self.im_width = conf.ds_params.im_width
        self.c_dim = conf.ds_params.c_dim
        #self.final_filters = conf.final_filters

        n = conf.ds_params.n
        m = conf.m
        self.learning_rate = conf.learning_rate
        self.beta1 = conf.beta1

        self.net_vars = []
        np.random.seed(0)  # BADD
        self.z = tf.placeholder(tf.float32, shape=[None, conf.d], name="z_inputs")
        self.x = tf.placeholder(tf.float32, shape=[None, self.im_height, self.im_width, self.c_dim], name="images")

        #Build the generator network
        s_h, s_w = self.im_height, self.im_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

        # Project and reshape
        z1, layer_vars = linear(self.z, self.im_height * 16 * s_h8 * s_w8, conf.d, 'g_h0_lin', with_w=True)
        self.net_vars += layer_vars


        self.h0 = tf.reshape(
            z1, [-1, s_h8, s_w8, self.im_height * 16])
        g_bn0 = batch_norm(name='g_bn0')
        h0 = tf.nn.relu(g_bn0(self.h0))

        out_1 = self.im_height * 8
        m = tf.shape(self.z)[0]
        self.h1, layer_vars = deconv2d(
            h0, tf.stack([m, s_h4, s_w4, out_1]), out_1, name='g_h1', with_w=True)
        g_bn1 = batch_norm(name='g_bn1')
        h1 = tf.nn.relu(g_bn1(self.h1))
        self.net_vars += layer_vars

        h1a, layer_vars = deconv2d(
            h1, tf.stack([m, s_h4, s_w4, out_1]), out_1, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h1a', with_w=True)
        g_bn1a = batch_norm(name='g_bn1a')
        h1a = tf.nn.relu(g_bn1a(h1a))
        self.net_vars += layer_vars

        out_2 = self.im_height * 4
        h2, layer_vars = deconv2d(
            h1a, tf.stack([m, s_h2, s_w2, out_2]), out_2, name='g_h2', with_w=True) #h1 replaced with h1a
        self.net_vars += layer_vars
        g_bn2 = batch_norm(name='g_bn2')
        h2 = tf.nn.relu(g_bn2(h2))

        h2a, layer_vars = deconv2d(
            h2, tf.stack([m, s_h2, s_w2, out_2]), out_2, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h2a', with_w=True)
        self.net_vars += layer_vars
        g_bn2a = batch_norm(name='g_bn2a')
        h2a = tf.nn.relu(g_bn2a(h2a))

        h3, layer_vars = deconv2d(
             h2a, [m, s_h, s_w, self.c_dim], self.c_dim, name='g_h3', with_w=True) #h2 replaced with h2a
        self.output = tf.nn.tanh(h3)
        self.net_vars += layer_vars

        #self.output = tf.nn.tanh(h3)

        mse_loss = tf.reduce_mean(tf.squared_difference(self.output, self.x), axis=[0, 3])
        self.mse_loss = tf.multiply(0.5, mse_loss)

    def train_net(self, learning_rate, beta1, global_step=None):
        """
        Get the gradients from the MS reconstruction error for the batch (i.e don't apply
        them just yet).
        :param z_batch: 
        :return: 
        """
        #For training the network parameters
        #network_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
        network_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

        #Compute the gradients for the network
        net_grads = tf.gradients(self.mse_loss, self.net_vars)

        #Apply the gradients for the network
        apply_op = network_opt.apply_gradients(zip(net_grads, self.net_vars), global_step=global_step)

        return apply_op, self.mse_loss

    def train_z(self):
        return tf.transpose(tf.gradients(self.mse_loss, [self.z])[0])

    def train_both(self, learning_rate, beta1, global_step=None):
        train_net, mse = self.train_net(learning_rate, beta1, global_step)
        return train_net, mse, self.train_z()

    def feed_forward(self):
        return self.output

    def gen_image(self):
        m = tf.shape(self.output)[0]
        return tf.reshape(self.output, [m, self.im_height, self.im_width, self.c_dim])


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))