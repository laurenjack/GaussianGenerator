import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from scipy.misc import imsave, imshow

#targrt_dict = {1: 'b', 2:'r', 3: 'g', 4:'c', 5:'m', 6:'y', 7:'k', 8:'w'}

def get_animator(n, d, do_animate, Z, targets):
    """
    Get the tracker used to track latent values over time for animation.

    :param n - Number of samples (latent values)

    :param d - Dimensionality of the latent space (must be 2 for animation
    to work)

    :param do_animate - Must be true for latent values to be tracked and
    animated.
    
    :param Z - The latent values for n images. NOTE, this object MUST be
    the same one used in the training process the animator is tracking
    (and stay that way throughout the animators life time). Futhermore,
    although the values of each latent vector might change, it is assumed
    that the latent vector at index i refers to the same image throughout
    the lifetime of the animator.
    
    :param targets: The targets for n images. Each index i MUST correspond
    to the latent value i in Z

    :return If n is too large (>2000), or the latent space is not 2D, or do_animate
    an inert NullAnimator will be returned (null object). Otherwise a LatentAnimator
    will be returned. which will store the value of the latents after each epoch and
    animate them after training.
    """
    if n <= 5000 and d == 2 and do_animate:
        return LatentAnimator(Z, targets)
    else:
        return NullAnimator()


class LatentAnimator:
    """A class responsible for tracking the latent values for each data
    point (and their target class) after each epoch and animating their
    movement post training"""

    def __init__(self, Z, scalar_targets):
        self.Z = Z
        n = Z.shape[1]
        #Get unique targets
        self.target_set = self._get_unique(scalar_targets)
        num_targets = len(self.target_set)

        #Create map of targets to indices in Z which have that target:
        self.targ_to_indices = {t:[] for t in self.target_set}
        for i in xrange(n):
            t = scalar_targets[i]
            self.targ_to_indices[t].append(i)

        #Create map of targets to colours so they can be identified on
        #a scatter plot
        color_func = get_cmap(num_targets)
        self.target_to_color = {t: color_func(j) for t, j in zip(self.target_set, range(num_targets))}

        #Create a list to hold each the target to latent map at the end
        #created at the end of each epoch
        self.all_prev_latents = []
        self.record_update()


    def record_update(self):
        """
        Add the new values for Z to the list that tracks Z over time.
        """
        target_to_latents = {t: self.Z[:, self.targ_to_indices[t]] for t in self.target_set}
        self.all_prev_latents.append(target_to_latents)

    def animate(self):
        """
        Animate the path of the latent variables as they move
        through a 2D z-space.
        """
        fig, ax = plt.subplots()
        epochs = len(self.all_prev_latents)
        # ax.set_xlim(-3.0, 3.0)
        # ax.set_ylim(0.0, 0.5)
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
        current_latents = self.all_prev_latents[0]
        scat_dict = self._plot_all_scats(current_latents, plt)
        scats = scat_dict.values()

        def init():
            return scats

        def update(frame):
            """Call back used by the animation to access the
            appropriate latent at the appropriate time. Each frame
            corresponds to an epoch of training."""
            current_latents = self.all_prev_latents[frame + 1]

            for t in self.target_set:
                zs = current_latents[t]
                num_t = zs.shape[1]
                as_list = np.split(zs, num_t, axis=1)
                scat_dict[t].set_offsets(as_list)
            return scats

        ani = FuncAnimation(fig, update, frames=epochs - 1, init_func=init, interval=500, repeat=False)

        plt.show()

    def _get_unique(self, scalar_targets):
        targ_set = set()
        for i in xrange(scalar_targets.shape[0]):
            t = scalar_targets[i]
            targ_set.add(t)
        return targ_set

    def _plot_all_scats(self, target_to_latents, plot):
        scat_dict = {}
        for t in self.target_set:
            zs = target_to_latents[t]
            xs = zs[0]
            ys = zs[1]
            scat = plot.scatter(xs, ys, color=self.target_to_color[t])
            scat_dict[t] = scat
        return scat_dict

    def gen_n_samples(self, sess, generator, n):
        z = np.random.randn(n, generator.d)
        images = sess.run(generator.gen_image(), feed_dict={generator.z : z})
        for i in xrange(n):
            imshow(images[i])

    def generate_30_samples_in_line_n_times(self, sess, generator, n):
        final_latents = self.all_prev_latents[-1]
        for i in xrange(n):
            self.generate_30_samples_in_line(sess, generator, final_latents)

    def generate_30_samples_in_line(self, sess, generator, final_latents):
        # Create the random direction which samples will be drawn from
        dir = np.random.randn(generator.d, 1)
        dir = dir / np.linalg.norm(dir)
        # Prepare to draw samples evenly along this line
        cs = np.arange(-2.0, 2.0, 4.0 / float(30))
        # Show distribution of Z's with the random direction shown as line
        plt.figure(1)
        ax = plt.gca()
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
        self.plot_latent_dist(plt, final_latents)
        xs = [dir[0, 0] * c for c in cs]
        ys = [dir[1, 0] * c for c in cs]
        plt.plot(xs, ys, color='k')

        # Next figure, show 20 observations along that line
        plt.figure(2)
        sub_num = 1
        for c in cs:
            z = c * dir
            image = sess.run(generator.gen_image(), feed_dict={generator.z : z.transpose()})[0]
            sub_plot = plt.subplot(5, 6, sub_num)
            sub_plot.yaxis.set_visible(False)
            sub_plot.xaxis.set_visible(False)
            sub_plot.imshow(image, interpolation='nearest')
            sub_num += 1

        plt.show()

    def plot_latent_dist(self, plot, final_latents):
        self._plot_all_scats(final_latents, plot)

class NullAnimator:
    """
    Null object, used when training a Gaussian Generator when we do
    not want to animate or store the latent values over time, but
    inert behavior instead. 
    """

    def record_update(self):
        pass

    def animate(self):
        pass

    def generate_30_samples_in_line_n_times(self, sess, generator, n):
        pass

    def gen_n_samples(self, sess, generator, n):
        z = np.random.randn(n, generator.d)
        images = sess.run(generator.gen_image(), feed_dict={generator.z : z})
        for i in xrange(n):
            imshow(images[i])

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='gist_rainbow')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color



