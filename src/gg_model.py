import numpy as np
import tensorflow as tf
import math
import data_loader
from ops import *

class GaussianGenerator:

    def __init__(self, sess, n, d, data_dir, config, r=30, m=64, final_filters=128, im_height=32, im_width=32, c_dim=3):
        self.sess = sess
        self.n = n
        self.d = d
        self.r = r
        self.m = m
        self.im_height = im_height
        self.im_width = im_width
        self.c_dim = c_dim
        self.final_filters = final_filters
        self.h = (4.0 / (3.0 * n)) ** 0.2
        self.h_c = tf.constant(self.h, name = "h")
        self.inv_root_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        self.neg_half = tf.constant(-1.0 / 2.0)

        # Create the latent value tensors, these hold the latent value for every data point
        z_values = tf.Variable(np.random(self.n, self.d), "Latent values")

        #Create f(z) to hold onto

        # Define the operations that load the data into batches
        ids, images, labels = data_loader.inputs(data_dir, m)

        # Get the latent z's that correspond to this batch of images
        zt = tf.gather(z_values, ids)

        #Reshape and project to a conv architecture
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            #Project and reshape
            z_, self.h0_w, self.h0_b = linear(zt, self.final_filters * 4 * s_h8 * s_w8, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                z_, [-1, s_h8, s_w8, self.final_filters * 4])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h4, s_w4, self.final_filters * 2], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.final_filters * 1], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.c_dim], name='g_h3', with_w=True)

            self.generator = tf.nn.tanh(h3)

        mse_loss = tf.reduce_mean(tf.squared_difference(self.generator, images))

        network_opt = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        #Compute the gradients for the network, but don't update yet
        grads = network_opt.compute_gradients(mse_loss)

        # Apply gradients, for the network, and the latent z's
        apply_gradient_op = network_opt.apply_gradients(grads)



        x = tf.placeholder(tf.float32, (m, im_height, im_width, c_dim), 'Image targets')

        #Create the regularizer

        #Stick with standard NN convention for mat muls i.e. Wz rather than zTWT
        z = tf.transpose(zt)

        #Tensor for the reference samples
        z_ref_t = tf.placeholder(tf.float32, (r, d), "References")
        D = tf.placeholder(tf.float32, (d, d), "Random directions")

    def f(self, z_values, m, n, d):
        """Compute f(z), which is calculated across all data points"""
        #Figure out the number of z batches, and the size of the last batch
        num__full_batches = n // m
        last_batch_size = n % m

        #Compute f(z) batch-wise (order doesn't matter, its deterministic
        #given the set of references
        for b_start in xrange(num__full_batches):
            zt = tf.slice(z_values, [b_start, 0], [m, d])

    def f_for_batch(self):
        pass


    def graph_not_sure(self, z, z_ref_t, D):
        """
        Computes: the contribution to f(z) a given batch of
        :return:
        1) kernel sum for batch -the contribution the batch of z's
        makes to the total f(z)
        2) df/dz for the given batch (must be stored)
        """
        distance_along_dirs, distances_squared, kernels, kernel_sum = self.fz_helper(z, z_ref_t, D)
        df_dz = tf.divide(distance_along_dirs, self.h_c)
        soften_exp = tf.add(tf.constant(1.0), tf.sqrt(distances_squared))
        df_dz = tf.divide(df_dz, soften_exp)
        return kernel_sum, df_dz

    def graph_not_sure_either(self, fz, z_fed, z_var, z_ref, D):
        """

        :param fz:
        :param df_dz:
        :param z:
        :param refs:
        :param D:
        :return:
        """
        #Compute the gradient for z and update it
        #p(z)
        pz = self.p(z_ref, D)
        #f(z) - p(z)
        error = tf.reshape(tf.subtract(fz, pz), [self.r, self.d, 1])

        #f'(z)
        distance_along_dirs, distances_squared, _, _ = self.fz_helper(z_fed, z_ref, D)
        df_dz = tf.divide(distance_along_dirs, self.h_c)
        soften_exp = tf.add(tf.constant(1.0), tf.sqrt(distances_squared))
        df_dz = tf.divide(df_dz, soften_exp)

        # f'(z)(f(z) - p(z))
        dL_dz = tf.reduce_sum(tf.multiply(df_dz, error), 0)
        dL_dz = tf.divide(tf.matmul(D, dL_dz), tf.constant(float(self.r)))
        z_lambda = tf.placeholder(tf.float32, name="z learning rate")
        delta_z = tf.multiply(z_lambda, dL_dz)
        z_var = tf.plus(z_var, delta_z)

        #Update f(z)
        _, _, _, new_ks = self.fz_helper(z_var, z_ref, D)
        old_ks = tf.placeholder(tf.float32, name="Previous Kernel Sum")
        delta_ks = tf.subtract(new_ks, old_ks)
        fz = tf.add(fz, delta_ks)
        return z_var, fz


    def fz_helper(self, z, z_ref, D):
        """Operation that produces the tensors required to compute/update f(z),
        or f'(z)
        """
        z_ref_t = tf.transpose(z_ref)
        # Resize both to consider all z - z_ref in a 3-tensor
        z_tensor = tf.resize(z, [1, self.d, self.m])
        z_ref_tensor = tf.resize(z_ref_t, [self.r, self.d, 1])
        z_tensor = tf.tile(z_tensor, [self.r, 1, 1])
        z_ref_tensor = tf.tile(z_ref_tensor, [1, 1, self.m])

        # Now find difference of z's
        z_diff = tf.subtract(z_ref_tensor, z_tensor)

        # Figure out how far each difference is projected along each direction
        Dt = tf.transpose(D)
        distance_along_dirs = tf.matmul(Dt, z_diff)
        distance_along_dirs = tf.divide(distance_along_dirs, self.h_c)
        distances_squared = tf.square(distance_along_dirs)

        # Compute the Gaussian kernel for each difference, in each direction
        kernels = tf.exp(tf.mul(self.neg_half, distances_squared))
        c = tf.constant(self.inv_root_2pi / (self.h * float(self.n)))
        kernels = tf.mul(c, kernels)
        kernel_sum = tf.reduce_sum(kernels, 2)
        return distance_along_dirs, distances_squared, kernels, kernel_sum


    def p(self, z_ref, D):
        Dt = tf.transpose(D)
        dists_sq = tf.square(tf.mat_mul(Dt, z_ref))
        exponent = tf.mul(self.neg_half, dists_sq)
        norm_dist = tf.mul(self.inv_root_2pi, tf.exp(exponent))
        return norm_dist.transpose()



    def train(self, num_epochs, data_dir, m, config):

        tf.global_variables_initializer().run()

        #Define any sumaries here

        for epoch in xrange(num_epochs):



def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))



foo = [2, 1] + [3, 4, 5]
print foo