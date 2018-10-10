import math
import numpy as np
from numpy.linalg import inv

def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    if _support_singular:
        return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
    return multivariate_normal.logpdf(flat_x, flat_mean, cov)

def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.

    Parameters:
    
    P : nd.array shape (2,2)
       covariance matrix

    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.

    Returns (angle_radians, width_radius, height_radius)
    """
    U, s, _ = linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)

def plot_covariance_ellipse(
        mean, cov=None, variance=1.0, std=None,
        ellipse=None, title=None, axis_equal=True, show_semiaxis=False,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):

    warnings.warn("deprecated, use plot_covariance instead", DeprecationWarning)
    plot_covariance(mean=mean, cov=cov, variance=variance, std=std,
                    ellipse=ellipse, title=title, axis_equal=axis_equal,
                    show_semiaxis=show_semiaxis, facecolor=facecolor,
                    edgecolor=edgecolor, fc=fc, ec=ec, alpha=alpha,
                    xlim=xlim, ylim=ylim, ls=ls)