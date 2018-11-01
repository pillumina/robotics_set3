from math import sin, cos, sqrt, atan2
import numpy as np
import matplotlib.pyplot as plt

def move(x, dt, u, l):
    """
    :param x:  (x, y, theta).T
    :param dt: delta t
    :param u:  (Vl, vr).T
    :param l:  length of vehicle
    :return:  updated states
    """
    hdg = x[2]
    vl = u[0]
    vr = u[1]
    R = (l / 2) * (vl + vr) / (vr - vl)
    w = (vr - vl) / l
    sinh, sinhb = sin(hdg), sin(w*dt)
    cosh, coshb = cos(hdg), cos(w*dt)

    return x + np.array([R*coshb*sinh + R*cosh*sinhb - R*sinh,
                         R*sinhb*sinh - R*cosh*coshb + R*cosh,
                         w*dt])


    # if the dt is small, then our estimate would give a reasonably accurate prediction.
    # I'll use this function to implement the state transition function f(x).

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x



