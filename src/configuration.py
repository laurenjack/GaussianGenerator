

class Conf:
    """
    Responsible for holding hyperparameters and data set specific params.
    
    Using this class isn't required, but used in conjunction with ConfBuilder
    it can catch missing assignments.
    """
    def __init__(self, builder):
        #Data set parameters
        self.ds_params = builder.ds_params

        #Training dimensions
        self.d = builder.d
        self.r = builder.r

        #Hyper-parameters
        self.m = builder.m
        self.num_epochs = builder.num_epochs
        self.solo_epochs = builder.solo_epochs
        self.z_eta = builder.z_eta
        self.reg_eta = builder.reg_eta
        self.learning_rate = builder.learning_rate
        self.beta1 = builder.beta1
        self.decay_rate = builder.decay_rate
        self.decay_epochs = builder.decay_epochs
        self.decay_z_epochs = builder.decay_z_epochs
        self.decay_reg_epochs = builder.decay_reg_epochs

        self.do_animate = builder.do_animate

    class Builder:
        pass


class DsParams:
    """
    Represents the parameters of an image data set
    """

    def __init__(self, n, im_height, im_width, c_dim):
        """
        Group the parameters of an image data set in a single object
        
        :param n - the number of images in the data set:
        :param im_height - the height of each image: 
        :param im_width - the width of each image: 
        :param c_dim - the number of colour channels of each image: 
        """
        self.n = n
        self.im_height = im_height
        self.im_width = im_width
        self.c_dim = c_dim
