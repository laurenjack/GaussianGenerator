import tensorflow as tf
import numpy as np
import math
from tensorflow.python import debug as tf_debug

class KdeRegularizer:

    def __init__(self, conf):
        #self.m = conf.m
        self.n = conf.ds_params.n
        self.d = conf.d
        self.r = conf.r

        self.z = tf.placeholder(tf.float32, (self.d, None), "Latent_Variables")
        self.z_ref = tf.placeholder(tf.float32, (self.d,self.r), "References")
        self.D = tf.placeholder(tf.float32, (self.d, self.d), "Random_directions")
        self.fz = tf.placeholder(tf.float32, (self.r,self.d), "fz")
        self.old_ks = tf.placeholder(tf.float32, name="old_fz_for_batch")

        self.h = (4.0 / (3.0 * self.n)) ** 0.2
        self.h_c = tf.constant(self.h, dtype=tf.float32, name = "h")
        self.inv_root_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        self.neg_half = tf.constant(-1.0 / 2.0, dtype=tf.float32)

    def fz_for_batch(self):
        """Operation that produces the tensors required to compute/update f(z)
        """
        distance_along_dirs, distances_squared = self.dfz_helper()

        # Compute the Gaussian kernel for each difference, in each direction
        kernels = tf.exp(tf.multiply(self.neg_half, distances_squared), name="exponentiated_dists")
        c = tf.constant(self.inv_root_2pi / (self.h * float(self.n)), dtype=tf.float32)
        kernels = tf.multiply(c, kernels, name='kernels')
        kernel_sum = tf.reduce_sum(kernels, 2, name="kernel_sum")
        return kernel_sum

    def dK_dz(self):
        """

        """
        D = self.D
        fz = self.fz

        # p(z)
        pz = self.p()
        #f(z) - p(z)
        error = tf.reshape(tf.subtract(fz, pz), [self.r, self.d, 1])

        #f'(z)
        distance_along_dirs, distances_squared = self.dfz_helper()
        df_dz = tf.divide(distance_along_dirs, self.h_c)
        soften_exp = tf.add(tf.constant(1.0), tf.sqrt(distances_squared))
        df_dz = tf.divide(df_dz, soften_exp)

        # f'(z)(f(z) - p(z))
        dK_dz = tf.reduce_sum(tf.multiply(df_dz, error), 0)
        return tf.divide(tf.matmul(D, dK_dz), tf.constant(float(self.r)))

    def update_fz(self):
        """
        Given a new value for the batch z, update f(z) to reflect these
        new z values.
           """
        new_ks = self.fz_for_batch()
        delta_ks = tf.subtract(new_ks, self.old_ks)
        return tf.add(self.fz, delta_ks)

    def dfz_helper(self):
        """Operation that produces the tensors required to compute f'(z)
        """
        z_ref_t = tf.transpose(self.z_ref)
        # Resize both to consider all z - z_ref in a 3-tensor
        z_tensor = tf.reshape(self.z, [1, self.d, -1])
        z_ref_tensor = tf.reshape(z_ref_t, [self.r, self.d, 1])
        z_tensor = tf.tile(z_tensor, [self.r, 1, 1])
        z_ref_tensor = tf.tile(z_ref_tensor, [1, 1, tf.shape(self.z)[1]])

        # Now find difference of z's
        z_diff = tf.subtract(z_ref_tensor, z_tensor)

        # Figure out how far each difference is projected along each direction
        Dt = tf.transpose(self.D)
        Dt = tf.reshape(Dt, [1, self.d, self.d])
        Dt = tf.tile(Dt, [self.r, 1, 1])
        distance_along_dirs = tf.matmul(Dt, z_diff)
        distance_along_dirs = tf.divide(distance_along_dirs, self.h_c)
        distances_squared = tf.square(distance_along_dirs)

        return distance_along_dirs, distances_squared

    def p(self):
        Dt = tf.transpose(self.D)
        dists_sq = tf.square(tf.matmul(Dt, self.z_ref))
        exponent = tf.multiply(self.neg_half, dists_sq)
        norm_dist = tf.multiply(self.inv_root_2pi, tf.exp(exponent))
        return tf.transpose(norm_dist)


# class Config():
#     pass
#
# config = Config()
# config.n = 60000
# config.d = 20
# config.m = 60000
# config.r = 30



# z_ref_init = np.random.randn(config.d, config.r).astype(np.float32)
# D_init = rvs(config.d).astype(np.float32)
# z_init = np.random.randn(config.n, config.d).astype(np.float32)
# start = datetime.now()
# reg = KdeRegularizer(config)
# train_op, z, z_ref, D = reg.train()
# sess = tf.Session()
# # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.run(tf.global_variables_initializer())
# fz_tf = np.zeros((config.r, config.d))
# batches = [z_init[k:k + config.m].transpose() for k in xrange(0, config.n, config.m)]
# for batch in batches:
#     kernel_sum = sess.run(train_op, feed_dict={z:batch, z_ref:z_ref_init, D:D_init})
#     fz_tf += kernel_sum
# end = datetime.now()
# print (end - start).total_seconds()
# import tst_kde as test_against
# test_against.speed_test(config, z_init, z_ref_init, D_init, fz_tf)


