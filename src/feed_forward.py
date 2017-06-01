import tensorflow as tf
import numpy as np

class FeedForward:
    """Feedforward generator network"""

    def __init__(self, conf):
        self.d = conf.d
        self.im_height = conf.ds_params.im_height
        self.im_width = conf.ds_params.im_width
        c_dim = conf.ds_params.c_dim
        num_pixel = self.im_height * self.im_width * c_dim
        self.learning_rate = conf.learning_rate

        self.net_vars = []
        np.random.seed(0) #BADD
        self.z = tf.placeholder(tf.float32, shape=[None, conf.d], name="z_inputs")
        self.x = tf.placeholder(tf.float32, shape=[None, num_pixel], name="images")

        a1, self.W1 = ff_layer(self.z, 784, "W1", "b1", "layer_1_out", self.net_vars)
        a2, _ = ff_layer(a1, 784, "W2", "b2", "layer_2_out", self.net_vars)
        self.output, _ = ff_layer(a2, num_pixel, "W3", "b3", "final_output_layer", self.net_vars, act_string='sig')

        #Setup loss to train on
        #mse_loss = tf.reduce_mean(tf.squared_difference(self.output, self.x), axis=0, name="Mean_Reconstruction_Error_DUB")
        sq_diff = tf.squared_difference(self.output, self.x)
        m = tf.cast(tf.shape(self.x)[0], dtype=tf.float32)
        self.mse_loss = tf.multiply(tf.divide(0.5, m), sq_diff, name="Mean_Reconstruction_Error")

    def train_net(self):
        #Create sgd optimizer
        relu_sgd = tf.train.GradientDescentOptimizer(0.01)
        sig_sgd = tf.train.GradientDescentOptimizer(0.1)

        #Compute gradients
        w_grads = tf.gradients(self.mse_loss, self.net_vars)
        #dL_dz = tf.gradients(mse_loss, self.z)[0]

        #Update the network
        train_sig = sig_sgd.apply_gradients(zip(w_grads[4:6], self.net_vars[4:6]))
        train_relu = relu_sgd.apply_gradients(zip(w_grads[0:4], self.net_vars[0:4]))

        return train_sig, train_relu, self.mse_loss

    def train_z(self):
        return tf.transpose(tf.gradients(self.mse_loss, self.z)[0])

    def train_both(self):
        train_sig, train_relu, mse = self.train_net()
        return train_sig, train_relu, mse, self.train_z()

    def feed_forward(self):
        return self.output

    def gen_image(self):
        m = tf.shape(self.output)[0]
        return tf.reshape(self.output, [m, self.im_height, self.im_width])


def ff_layer(input, num_output, w_name, b_name, out_name, net_vars, act_string='relu'):
    num_input = input.get_shape()[1].value
    act, w_init, b_init = act_and_weights(act_string, num_input, num_output)
    W = tf.Variable(w_init, name=w_name)
    b = tf.Variable(b_init, name=b_name)
    phi1 = tf.nn.bias_add(tf.matmul(input, W), b)
    net_vars.append(W)
    net_vars.append(b)
    return act(phi1, name=out_name), W

def act_and_weights(act_string, in_d, out_d):
    xc = 1.0 / float(in_d) ** 0.5
    if act_string == 'relu':
        act = tf.nn.relu
        weights = xc * np.random.randn(in_d, out_d)
        biases = 0.1 * np.random.randn(out_d)
    elif act_string == 'sig':
        act = tf.nn.sigmoid
        weights = 16 * xc * np.random.randn(in_d, out_d)
        biases = 16 * xc * np.random.randn(out_d)
    return act, weights.astype(np.float32), biases.astype(np.float32)
