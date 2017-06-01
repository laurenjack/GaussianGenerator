from configuration import *
from gaussian_generator import GaussianGenerator
import data_loader as dl
import numpy as np
from tensorflow.python import debug as tf_debug

# import sys
# sys.path.insert(1, "/home/laurenjack/PycharmProjects/ModularNN")
# sys.path.append("/home/laurenjack/PycharmProjects/ModularNN/data")
#
# import numpy as np
#
# from srcNN.generative import input_generator as ig
# from srcNN.generative import experiment_gg as py_gg

import math

def train__gg():
    #Load the data and data set parameters
    #images, labels, ds_params = dl.load_mnist(specific_targets=[3,6], sub_n=1000, feed_forward=False)
    filenames = ['/home/laurenjack/data/cifar-10-batches-bin/data_batch_'+str(i)+'.bin' for i in xrange(1, 6)]
    #cifar10_batch = '/home/laurenjack/data/cifar-10-batches-bin/data_batch_1.bin'
    images, labels, ds_params = dl.load_single_cifar_10_file(filenames, specific_targets=[3], sub_n=4800)
    #D, z_refs, Z, images_py, labels_py, n = ig.gen_stochastic_inputs(2, 100, file_path='/home/laurenjack/PycharmProjects/ModularNN/srcNN/data/mnist.pkl.gz')

    # #Now use same data for tensorflow version
    # ds_params = DsParams(n, 28, 28, 1)
    # images = np.concatenate(images_py, axis=1).transpose().astype(np.float32)
    # labels = np.array(labels_py).astype(np.float32)
    # print n
    
    #Build the configuration object
    b = Conf.Builder()
    b.ds_params = ds_params
    # Training dimensions
    b.d = 10
    b.r = 100
    # Hyper-parameters
    b.m = 50
    b.num_epochs = 250
    b.solo_epochs = 100
    b.z_eta = 0.01 #0.003
    b.reg_eta = 0.05 #1.0 #* 0.000001
    b.learning_rate = 0.001 #0.0005
    b.decay_rate = 0.6
    b.decay_epochs = 45
    b.decay_z_epochs = 30
    b.decay_reg_epochs = 45
    b.beta1 = 0.5
    b.do_animate = True
    conf = Conf(b)

    #Create the GG and train
    gg = GaussianGenerator(conf)
    #Run the other guy first
    #py_gg.run_gg(D, z_refs, Z, images_py, labels_py, gg)
    gg.train(images, labels)


if __name__ == '__main__':
    train__gg()