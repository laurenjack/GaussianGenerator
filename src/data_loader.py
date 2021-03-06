import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as in_mnist
import numpy as np
import os
from configuration import DsParams
import struct
from matplotlib import pyplot as plt
from scipy.misc import imsave, imshow

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def load_mnist(combine=False, specific_targets=None, sub_n=None, feed_forward=False):
    """Load the MNIST data, and data set params
    
    :param combine - If true will will return S = train U test U validation,
    (or a sample of S if sub_n is specified) otherwise will return train,
    (or sample of train)
    
    :param specific_targets - If specified, draw images which are only of the
    classes in this list
    
    :param sub_n - If specified, will return a sample of the constructed data set.
     
    :return The data set to be used for training, and the parameters
    associated with it.
    """
    mnist = in_mnist.read_data_sets('MNIST_data', one_hot=False)
    images = mnist.train.images
    labels = mnist.train.labels

    # Combine test, training and val (if combine is true)
    if combine:
        test = mnist.test
        vald = mnist.validation
        images = np.concatenate((images, test.images, vald.images), axis=0)
        labels = np.concatenate((labels, test.labels, vald.labels))

    images, labels, n = _select_data(images, labels, specific_targets, sub_n)

    #Build DsParams object
    dsParams = DsParams(n, 28, 28, 1)
    if feed_forward:
        return images, labels, dsParams
    return images.reshape(n, 28, 28, 1), labels, dsParams

def load_single_cifar_10_file(filenames, specific_targets=None, sub_n=None):
    #CIFAR 10 data set properties
    im_height = 32
    im_width = 32
    c_dim = 3
    num_bytes = im_height * im_width * c_dim
    n = 10000

    #Load raw data and split into labels and images
    all_bytes = []
    for filename in filenames:
        all_byte = np.fromfile(filename, 'uint8')
        all_byte = all_byte.reshape(n, num_bytes + 1)
        all_bytes.append(all_byte)
    all_byte = np.concatenate(all_bytes, axis=0)
    labels = all_byte[:, 0]
    images = all_byte[:, 1:]
    images, labels, n = _select_data(images, labels, specific_targets, sub_n)

    #Arrange images according to the networks layout
    images = images.reshape(n, c_dim, im_height, im_width)
    images = images.transpose([0,2,3,1])
    im_int = images

    #Standardize each image individually
    images = images.astype(np.float32)
    max = np.max(images)
    min = np.min(images)
    # maxes = images.max(axis=(1, 2, 3)).reshape(n, 1, 1, 1)
    # mins = images.min(axis=(1, 2, 3)).reshape(n, 1, 1, 1)
    pix_range = max - min
    images = (images - min) / pix_range


    # im_means = np.mean(images, axis=(1,2,3)).reshape(n,1,1,1)
    # sds = (1.0/float(num_bytes)*np.sum((images - im_means) ** 2.0, axis=(1,2,3))) ** 0.5
    # sds = sds.reshape(n,1,1,1)
    # images = (images - im_means) / sds

    # Build DsParams object
    ds_params = DsParams(n, im_height, im_width, c_dim)
    return images, labels, ds_params

def _save_image(image, filename):
    #image = _inverse_transform(im_net)
    imshow(image)
    #imsave(filename, image)

def _inverse_transform(im_net, filename):
  return (im_net+1.0)/2.0


def _select_data(images, labels, specific_targets, sub_n):
    n = images.shape[0]
    indices = np.arange(0, n)

    # If specifed, draw only those indices from specific classes
    if specific_targets is not None:
        indices = []
        # TODO make more pythonic
        for i in xrange(n):
            if labels[i] in specific_targets:
                indices.append(i)
        n = len(indices)
        indices = np.array(indices)

    # Take random sample of size sub_n (if specified)
    if sub_n is not None:
        indices = np.random.choice(indices, size=sub_n, replace=False)
        n = sub_n

    images = images[indices]
    labels = labels[indices]

    return images, labels, n



class ReaderWriter:

    def __init__(self, x_size, d):
        self.x_size = x_size
        self.d = d

    def load_x_z_df(self, filename_queue):
        reader = tf.TextLineReader()
        key, line = reader.read(filename_queue)
        label, x = self.read_cifar10(filename_queue)
        defaults = [-1.0] + [0.0] * (self.x_size + 2 * self.d)
        values = tf.decode_csv(line, record_defaults=defaults)
        label = values[0]
        start_z = self.x_size + 1
        x = values[1:start_z]
        start_df = start_z + self.d
        z = values[start_z: start_df]
        end_df = start_df + self.d
        df_dz = values[start_df:end_df]
        return x, z, df_dz

    def input_queue(self, filenames, m):
        filename_queue = tf.train.string_input_producer(filenames)
        x, z, df_dz = self.load_x_z_df(filename_queue)
        min_after_dequeue = 10000
        cap = min_after_dequeue + 3 * m
        x_batch, z_batch, dfdz_batch = tf.train.shuffle_batch(
            [x, z, df_dz], batch_size=m, capacity=cap,
            min_after_dequeue=min_after_dequeue)
        return x_batch, z_batch, dfdz_batch

    def _as_csv(self, prefix, var):
        return prefix+" "+var+".csv"


def inputs(data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    ID, image, label = read_cifar10(filename_queue)
    reshaped_image = tf.cast(image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(ID, float_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def _generate_image_and_label_batch(ID, image, label, min_queue_examples,
                                        batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        ids, images, label_batch = tf.train.shuffle_batch(
            [ID, image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ids, images, label_batch = tf.train.batch(
            [ID, image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return ids, images, tf.reshape(label_batch, [batch_size])

def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    Args:
      filename_queue: A queue of strings with the filenames to read from.
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    id_bytes = 4
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = id_bytes + label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    id = tf.cast(
        tf.strided_slice(record_bytes, [0], [id_bytes]), tf.int32)

    label = tf.cast(
        tf.strided_slice(record_bytes, [id_bytes], [id_bytes + label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [id_bytes + label_bytes],
                         [id_bytes + label_bytes + image_bytes]),
        [depth, height, width])
    # Convert from [depth, height, width] to [height, width, depth].
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    return id, uint8image, label




