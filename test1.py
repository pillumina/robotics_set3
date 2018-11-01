# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:07:20 2018
@author: johndoe
TO DO:
When done debugging, convert phi as control variable to voltage.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_results(s, u):
    plt.subplot(1, 2, 2)
    plt.plot(s[0, :], s[1, :], '--')

    color = 'tab:grey'
    linestyle = 'dashed'
    plt.plot([-250E-3, 250E-3], [375E-3, 375E-3], color=color, linestyle=linestyle)
    plt.plot([250E-3, 250E-3], [375E-3, -375E-3], color=color, linestyle=linestyle)
    plt.plot([-250E-3, 250E-3], [-375E-3, -375E-3], color=color, linestyle=linestyle)
    plt.plot([-250E-3, -250E-3], [-375E-3, 375E-3], color=color, linestyle=linestyle)

    xmin = np.min([np.min(s[0, :]), -250E-3])
    xmax = np.max([np.max(s[0, :]), 250E-3])
    ymin = np.min([np.min(s[1, :]), -375E-3])
    ymax = np.max([np.max(s[1, :]), 375E-3])
    k = 1.2
    plt.xlim(k * xmin, k * xmax)
    plt.ylim(k * ymin, k * ymax)

    xvals = dt * np.arange(0, np.shape(s)[1])
    plt.subplot(2, 2, 1)
    plt.plot(xvals, s[0, :])
    plt.plot(xvals, s[1, :])
    plt.plot(xvals, s[2, :])
    plt.legend(['x position', 'y position', 'Heading'])

    xvals = np.arange(0, np.shape(u)[1])
    plt.subplot(2, 2, 3)
    plt.plot(xvals, u[0, :])
    plt.plot(xvals, u[1, :])
    plt.legend(['phi_dot_left', 'phi_dot_right'])

    plt.show()


def compose_B(s):
    accel_l = s[0, 0]
    accel_r = s[2, 0]
    theta = s[2, 0]

    F_ratio = accel_l / accel_r
    if accel_l == 0:
        y_cm = 0
    else:
        y_cm = 2 * delta * (1 - 1 / (F_ratio + 1))

    b0 = np.array([h * np.cos(theta), i * np.cos(theta)])
    b1 = np.array([h * np.sin(theta), i * np.sin(theta)])
    b2 = (1 / I_z) * np.array([[y_cm * m_l, (2 * delta - y_cm) * m_r]])

    B = dt * r * np.vstack((b0, b1, b2))

    return B


def omega(volts):
    #    j0 = ((130*2*np.pi)/60 - (100*2*np.pi)/60)/(6-4.8)

    # Assuming linear relationship passing through origin.
    j0 = (130 * 2 * np.pi / 60) / 6

    return j0 * volts


m_c = 0.3
m_r = 0.05
m_l = m_r
r = 20E-3  # 20 mm
delta = (85E-3) / 2  # 42.5 mm
dt = 0.1
I_z = m_r * r ** 2

phi_dot_l = np.concatenate((np.ones(200), -1 * np.ones(20), np.ones(200)), 0)
phi_dot_r = np.concatenate((np.ones(200), np.ones(20), np.ones(200)), 0)

u_hist = np.vstack((phi_dot_l, phi_dot_r))
s_hist = np.zeros([3, np.shape(u_hist)[1] + 1])

# rad/sec/volt # motor control gain

k0 = 1
k1 = 1
k2 = 1

x0, y0, theta0 = 0, 0, 0
s0 = np.array([[x0, y0, theta0]]).T

a0 = np.array([1, 0, 0])
a1 = np.array([0, 1, 0])
a2 = np.array([0, 0, 1])

A = np.vstack((a0, a1, a2))

h = m_l / m_c
i = m_r / m_c

B = compose_B(s0)

c0 = np.array([k0, 0, 0])
c1 = np.array([0, k1, 0])
c2 = np.array([0, 0, k2])

C = np.vstack((c0, c1, c2))

s_hist[:, 0:1] = s0
s = 1 * s0
u = 1 * u_hist[:, 0:1]

t_start = dt
t_stop = dt * np.shape(u_hist)[1] + dt
for n in range(1, np.shape(u_hist)[1] + 1):
    s = A @ s + B @ u

    # Update u
    u = u_hist[:, n:(n + 1)]
    B = compose_B(s)

    # Store s in state history
    s_hist[:, n:(n + 1)] = s

plot_results(s_hist, u_hist)