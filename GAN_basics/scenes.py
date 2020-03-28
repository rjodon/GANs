import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


def print_distribution(distri, equalAxes=True):
    """
    Prints a given distribution
    Parameters
    ----------
    distri np.array
        The given distribution to print.
    equalAxes bool
        Boolean to enable/disable equal axis.
    """
    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    hist = axs.hist2d(distri[:, 0], distri[:, 1], bins=100, density=True, norm=PowerNorm(gamma=1. / 2.))
    fig.colorbar(hist[3])
    if equalAxes:
        axs.axis('equal')


class SceneGaussian2D:
    """
    Generate a simple 2D gaussian distribution
    
    Attributes
    ----------
    mean {list, np.array}
        Contains the axial mean of the distribution (x and y mean).
    cov {list, np.array}
        The covariant matrix of the given distribution.
    are_polar_coordinates bool
        Defines if coordinates are polar or not.
    xy_samples np.array
        The distribution in cartesian coordinates.
    polar_samples np.array
        The distribution in polar coordinates.
    _cplx_samples {generator, None}
        A generator containing the distribution in complex representation. Used for conversion cart<->polar.
    """
    def __init__(self, mean, cov, nsamples, are_polar_coordinates=False):
        """
        Parameters
        ----------
        mean {list, np.array}
            Contains the axial mean of the distribution (x and y mean).
        cov {list, np.array}
            The covariant matrix of the given distribution.
        nsamples
        are_polar_coordinates bool
            Defines if coordinates are polar or not.
        """
        self.mean = mean
        self.cov = cov
        self.are_polar_coordinates = are_polar_coordinates
        self.xy_samples = self.generate_samples(nsamples)
        
        if are_polar_coordinates:
            self._cplx_samples = None  # Defined in cartesian to polar
            self.polar_samples = self.cartesian_to_polar()
    
    def generate_samples(self, samples):
        """
        Parameters
        ----------
        samples int
            The number of points in the distribution.s

        Returns np.array
        -------
            The distribution.
        """
        return np.random.multivariate_normal(self.mean, self.cov, samples)
    
    def cartesian_to_polar(self):
        """
        Returns np.array
        -------
            The distribution in cartesian coordinates
        """
        self._cplx_samples = (np.complex(s) for s in self.xy_samples)
        return np.array(np.absolute(self._cplx_samples), np.angle(self._cplx_samples)),
    
    def polar_to_cartesian(self):
        """
        Returns np.array
        -------
            The distribution in polar coordinates
        """
        return np.real(self._cplx_samples), np.imag(self._cplx_samples)
    
