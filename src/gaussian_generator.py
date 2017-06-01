import numpy as np
import tensorflow as tf
import utils
import latent_animator as la
from network import *
from feed_forward import *
from regularization import *
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

class GaussianGenerator:

    def __init__(self, conf):
        self.m = conf.m;
        self.n = conf.ds_params.n
        self.d = conf.d
        self.r = conf.r

        self.num_epochs = conf.num_epochs
        self.solo_epochs = conf.solo_epochs
        self.z_eta = conf.z_eta
        self.reg_eta = conf.reg_eta
        self.do_animate = conf.do_animate

        self.learning_rate = conf.learning_rate
        self.beta1 = conf.beta1
        self.decay_epochs = conf.decay_epochs
        self.decay_rate = conf.decay_rate
        self.decay_z_epochs = conf.decay_z_epochs
        self.decay_reg_epochs = conf.decay_reg_epochs

        self.G = Generator(conf)
        #self.G = FeedForward(conf)
        self.R = KdeRegularizer(conf)


    def train(self, images, labels):

        #Initialize the z_values
        Z = np.random.randn(self.d, self.n)

        #Keep iamges, labels and z's linked via indicies
        indices = np.array(range(0, self.n))

        #Create the animator (could be a null object, see latent_animator.py)
        animator = la.get_animator(self.n, self.d, self.do_animate, Z, labels)

        #Create global step variable for training just the network post z training
        global_step = tf.Variable(0, trainable=False)
        decay_steps = int(math.ceil(float(self.n) / float(self.m))) * self.decay_epochs
        decay_rate = self.decay_rate
        start_rate = self.learning_rate
        decaying_lr = tf.train.exponential_decay(start_rate, global_step, decay_steps, decay_rate, staircase=True)
        train_net = self.G.train_net(decaying_lr, self.beta1, global_step=global_step)

        # Get the operations required for training the Gaussian generator
        fz_for_batch_op, dK_dz_op, update_fz_op, gen_train_ops = self.__get_ops(self.learning_rate, self.beta1, global_step)

        # Start the tensorflow session
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())

        for epoch in xrange(self.num_epochs):
            #Create a set of references for regularization
            z_ref = np.random.randn(self.d, self.r)
            #Create d random orthogonal Directions
            D = utils.rvs(self.d)

            #Shuffle the data (via indicies) and get the new batches
            np.random.shuffle(indices)
            batches = self.__get_batches(indices)

            #Compute f(z) and kernel sums
            fz = np.zeros((self.r, self.d))
            kernel_sums = []
            for batch_inds in batches:
                z = Z[:, batch_inds]
                feed_dict = {self.R.z: z, self.R.z_ref: z_ref, self.R.D: D}
                kernel_sum = sess.run(fz_for_batch_op, feed_dict=feed_dict)
                kernel_sums.append(kernel_sum)
                fz += kernel_sum

            #Train the network, and the latents (while appling regularization to the latents)
            if (epoch + 1) % self.decay_z_epochs == 0:
                self.z_eta /= 2.0
            if (epoch + 1) % self.decay_reg_epochs == 0:
                self.reg_eta /= 2.0
            batch_num = 0
            mse = 0
            for batch_inds in batches:
                z = Z[:, batch_inds]
                x = images[batch_inds]
                # Calculate the gradients for z w.r.t to the network loss
                net_dict = {self.G.z:z.transpose(), self.G.x: x}
                _, mse_batch, dL_dz = sess.run(gen_train_ops, feed_dict=net_dict)
                #dL_dz = sess.run(z_train_ops, feed_dict=net_dict)
                #Make adjustment for reduce mean, which will make the grads to small by a factor of the batch_size
                dL_dz *= z.shape[1]
                #Calculate the gradients for z w.r.t the regularization loss
                reg_dict = {self.R.z: z, self.R.z_ref: z_ref, self.R.D: D, self.R.fz: fz}
                dK_dz = sess.run(dK_dz_op, feed_dict=reg_dict)

                #Update z using the gradients of both
                Z[:, batch_inds] -= (self.z_eta * dL_dz + self.reg_eta * dK_dz)

                #Update f(z) for the next batch
                #reg_dict[self.R.old_ks] = kernel_sums[batch_num]
                #fz = sess.run(update_fz_op, feed_dict=reg_dict)
                reg_dict[self.R.z] = Z[:, batch_inds]
                new_ks = sess.run(fz_for_batch_op, feed_dict=reg_dict)
                fz += new_ks - kernel_sums[batch_num]
                mse += np.sum(mse_batch)
                batch_num += 1

            print "Z Epoch "+str(epoch+1)+" Mse: "+str(mse)

            #Let the animator know the latent values have changed
            animator.record_update()

        #Now train just the network without moving the z's
        for epoch in xrange(self.solo_epochs):
            mse = 0
            np.random.shuffle(indices)
            for batch_inds in batches:
                z = Z[:, batch_inds]
                x = images[batch_inds]
                net_dict = {self.G.z: z.transpose(), self.G.x: x}
                _, mse_batch = sess.run(train_net, feed_dict=net_dict)
                mse += np.sum(mse_batch)
            print "Solo Epoch "+str(epoch+1)+" Mse: "+str(mse)

        #Perform the animation (unless the animator is a null object)
        animator.animate()
        animator.gen_n_samples(sess, self.G, 20)
        #animator.generate_30_samples_in_line_n_times(sess, self.G, 3)
        sess.close()


    def __get_ops(self, learning_rate, beta1, global_step=None):
        R = self.R
        G = self.G
        return R.fz_for_batch(), R.dK_dz(), R.update_fz(), G.train_both(learning_rate, beta1, global_step) #G.train_net(), G.train_z()

    def __get_batches(self, indices):
        return [indices[k:k + self.m] for k in xrange(0, self.n, self.m)]

    # if epoch % 2 == 0:
    #     dL_dz = sess.run(train_z_op, feed_dict=net_dict)
    #     dL_dz *= z.shape[1]
    #     # Calculate the gradients for z w.r.t the regularization loss
    #     reg_dict = {self.R.z: z, self.R.z_ref: z_ref, self.R.D: D, self.R.fz: fz}
    #     dK_dz = sess.run(dK_dz_op, feed_dict=reg_dict)
    #
    #     # Update z using the gradients of both
    #     Z[:, batch_inds] -= (self.z_eta * dL_dz + self.reg_eta * dK_dz)
    #
    #     # Update f(z)
    #     reg_dict[self.R.z] = Z[:, batch_inds]
    #     new_ks = sess.run(fz_for_batch_op, feed_dict=reg_dict)
    #     fz += new_ks - kernel_sums[batch_num]
    # else:
    #     _, mse_batch = sess.run(train_net_op, feed_dict=net_dict)
    #     mse += np.sum(mse_batch)



