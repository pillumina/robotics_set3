from __future__ import (absolute_import, division)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import cholesky
from Unscented_trans import unscented_transform
from helper import pretty_str



class UnscentedKalmanFilter(object):
    r"""
    Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier, using the formulation provided by Wan and Merle
    in. This filter scales the sigma points to avoid strong nonlinearities.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
        This is for convience, so everything is sized correctly on
        creation. If you are using multiple sensors the size of `z` can
        change based on the sensor. Just provide the appropriate hx function
    dt : float
        Time between steps in seconds.
    hx : function(x)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_z).
    fx : function(x,dt)
        function that returns the state x transformed by the
        state transistion function. dt is the time step in seconds.
    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. For example, MerweScaledSigmaPoints implements the alpha,
        beta, kappa parameterization of Van der Merwe, and
        JulierSigmaPoints implements Julier's original kappa
        parameterization. See either of those for the required
        signature of this class if you want to implement your own.
    sqrt_fn : callable(ndarray), default=None (implies scipy.linalg.cholesky)
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.
        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing as far as this class is concerned.
    x_mean_fn : callable  (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.
    z_mean_fn : callable  (sigma_points, weights), optional
        Same as x_mean_fn, except it is called for sigma points which
        form the measurements after being passed through hx().
    residual_x : callable (x, y), optional
    residual_z : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars. One is for the state variable,
        the other is for the measurement state.
    Attributes
    ----------
    x : numpy.array(dim_x)
        state estimate vector
    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix
    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : ndarray
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        process noise matrix
    K : numpy.array
        Kalman gain
    y : numpy.array
        innovation residual
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):


        #pylint: disable=too-many-arguments

        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty

        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '
        Important: this MUST be called before update() is called for the first
        time.
        Parameters
        ----------
        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.
        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.
        Parameters
        ----------
        z : numpy.array of shape (dim_z)
            measurement vector
        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.
        Parameters
        ----------
        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.
        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.
            If Rs is None then self.R is used for all epochs.
            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.
        dts : None, scalar or list-like, default=None
            optional value or list of delta time to be passed into predict.
            If dtss is None then self.dt is used for all epochs.
            If it is a list where len(dts) == len(zs), then it is treated as a
            list of dt values, one per epoch. This allows you to have varying
            epoch durations.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch
        Returns
        -------
        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.
        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        Examples
        --------
        .. code-block:: Python
            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts The output is then smoothed
            # with an RTS smoother.
            zs = [t + random.randn()*4 for t in range (40)]
            (mu, cov, _, _) = ukf.batch_filter(zs, dts=dts)
            (xs, Ps, Ks) = ukf.rts_smoother(mu, cov)
        """
        #pylint: disable=too-many-arguments

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not(isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError('zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return (means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.
        Parameters
        ----------
        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.
        Ps : numpy.array
            array of the covariances of the output of a kalman filter.
        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used
        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        Returns
        -------
        x : numpy.ndarray
           smoothed means
        P : numpy.ndarray
           smoothed state covariances
        K : numpy.ndarray
            smoother gain at each step
        Examples
        --------
        .. code-block:: Python
            zs = [t + random.randn()*4 for t in range (40)]
            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """
        #pylint: disable=too-many-locals, too-many-arguments

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))

        for k in reversed(range(n-1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q,
                self.x_mean, self.residual_x)

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, self.inv(Pb))

            # update the smoothed estimates
            xs[k] += dot(K, self.residual_x(xs[k+1], xb))
            ps[k] += dot(K, ps[k+1] - Pb).dot(K.T)
            Ks[k] = K

        return (xs, ps, Ks)


    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis
