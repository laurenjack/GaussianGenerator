import sys
sys.path.insert(1, "/home/jack/PycharmProjects/ModularNN")

import numpy as np
import tensorflow as tf
from datetime import datetime
from utils import *
from unittest import *
from regularization import KdeRegularizer
from srcNN.generative import kde_regularizer as modNN_reg

class Config():
    pass

config = Config()
config.n = 60000
config.d = 20
config.m = 128
config.r = 30
z_ref_init = np.random.randn(config.d, config.r).astype(np.float32)
D_init = rvs(config.d).astype(np.float32)
z_init = np.random.randn(config.n, config.d).astype(np.float32)
batches = [z_init[k:k + config.m].transpose() for k in xrange(0, config.n, config.m)]
reg = KdeRegularizer(config)
fz_for_batch_op = reg.fz_for_batch()

class KdeRegularizerSpec(TestCase):

    def test_fz(self):
        print "f(z)"
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = datetime.now()
        fz_tf, _ = self.__fz_tf(sess)
        end = datetime.now()
        print (end - start).total_seconds()
        #Test against the same regularization algorithm written purely in python
        start = datetime.now()
        density_stats = modNN_reg.f(batches, config.d, config.r, config.n, None, z_ref_init, D_init)
        fz = density_stats.fz
        end = datetime.now()
        print (end - start).total_seconds()
        np.testing.assert_array_almost_equal(fz_tf, fz)
        sess.close()

    def test_delta_z(self):
        print "delta_z"
        dK_dz_op = reg.dK_dz()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start = datetime.now()
        #Find fz so we may compute delta_z
        fz, _ = self.__fz_tf(sess)
        dK_dzs_tf = []
        for batch in batches:
            dK_dz = sess.run(dK_dz_op, feed_dict={reg.z: batch, reg.z_ref: z_ref_init, reg.D: D_init, reg.fz:fz})
            dK_dzs_tf.append(dK_dz)
        end = datetime.now()
        print (end - start).total_seconds()
        #Test against the same regularization algorithm written purely in python
        start = datetime.now()
        density_stats = modNN_reg.f(batches, config.d, config.r, config.n, None, z_ref_init, D_init)
        dK_dzs = []
        for bi in xrange(len(batches)):
            dK_dz = modNN_reg.get_grad_for(bi, density_stats)
            dK_dzs.append(dK_dz)
        end = datetime.now()
        print (end - start).total_seconds()
        for act, exp in zip(dK_dzs_tf, dK_dzs):
            np.testing.assert_array_almost_equal(act, exp)

    def test_update_z(self):
        print "Update f(z)"
        update_fz_op = reg.update_fz()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #Find fz so we have something to update
        fz, kernel_sums = self.__fz_tf(sess)
        updated_z = np.random.uniform(0, 1, size=(config.d, config.m))
        updated_fz_tf = sess.run(update_fz_op, feed_dict={reg.z: updated_z, reg.fz:fz, reg.z_ref: z_ref_init, reg.D: D_init, reg.old_ks:kernel_sums[0]})
        #Test against the same regularization algorithm written purely in python
        density_stats = modNN_reg.f(batches, config.d, config.r, config.n, None, z_ref_init, D_init)
        updated_fz = modNN_reg.update_fz(density_stats, updated_z, 0).fz
        np.testing.assert_array_almost_equal(updated_fz_tf, updated_fz)


    def __fz_tf(self, sess):
        fz_tf = np.zeros((config.r, config.d))
        kernel_sums = []
        for batch in batches:
            kernel_sum = sess.run(fz_for_batch_op, feed_dict={reg.z: batch, reg.z_ref: z_ref_init, reg.D: D_init})
            kernel_sums.append(kernel_sum)
            fz_tf += kernel_sum
        return fz_tf, kernel_sums


