from __future__ import division
import numpy as np
from scipy.linalg import cholesky
from helper import pretty_str

class MerweScaledSigmaPoints(object):

    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation for the UnscentedKalmanFilter class.. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.
    Parameters
    ----------
    n : int
        Dimensionality of the state. 2n+1 weights will be generated.
    alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3) according to [3].
    beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal, according to [3].
    kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 according to [4],
        or to 3-n according to [5].
    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.
    Attributes
    ----------
    Wm : np.array
        weight for each sigma point for the mean
    Wc : np.array
        weight for each sigma point for the covariance
    Examples
    --------
    """


    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
        Parameters
        ----------
        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
        Returns
        -------
        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)



    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('alpha', self.alpha),
            pretty_str('beta', self.beta),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
            ])
