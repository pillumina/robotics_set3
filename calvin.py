import numpy as np
from numpy.linalg import inv
import random
import matplotlib.pyplot as plt
import sympy as sy
from filterpy.stats import plot_covariance_ellipse

class Robot():
    def __init__(self):
        self.length = 750
        self.width = 500
        self.radius = 20
        self.H = np.eye(3)
        self.P = np.eye(3)
        self.max_motor_spd = np.pi
        self.motor_spd = 0.05 * self.max_motor_spd
        self.laser = 10
        self.imu = 0.25
        self.wheels_d = 85
        self.rr = 0.5
        self.noise = np.diag([self.laser ** 2, self.laser ** 2, self.imu ** 2])
        x_pos, y_pos, theta, l_spd, r_spd, radius, d, t = sy.symbols('x_pos, y_pos, theta, l_spd, r_spd, radius, d, t')
        self.func = self.symolize(x_pos, y_pos, theta, l_spd, r_spd, radius, d, t)
        self.state_dev = self.func.jacobian(sy.Matrix([x_pos, y_pos, theta]))
        self.input_dev = self.func.jacobian(sy.Matrix([l_spd, r_spd]))
        self.cur_state = np.array([0,500,7*np.pi/4])
        self.theta = self.cur_state[2]
        self.speed = np.array([0, 0])
        self.l_spd = l_spd
        self.r_spd = r_spd
        self.t = 0
        self.dic = self.init_dic(x_pos, y_pos, theta, l_spd, r_spd, radius, d, t)




    def symolize(self,x_pos, y_pos, theta, l_spd, r_spd, radius, d, t):
        func = sy.Matrix([[x_pos+1/2*sy.cos(theta)*(l_spd+r_spd)*radius*t], [y_pos+1/2*sy.sin(theta)*(l_spd+r_spd)*radius*t],
                          [theta+(r_spd-l_spd)*radius/d*t]])
        return func

    def init_dic(self, x_pos, y_pos, theta, l_spd, r_spd, radius, d, t):
        dic = {}
        dic[x_pos] = self.cur_state[0]
        dic[y_pos] = self.cur_state[1]
        dic[theta] = self.cur_state[2]
        dic[l_spd] = self.speed[0]
        dic[r_spd] = self.speed[1]
        dic[radius] = self.radius
        dic[d] = self.wheels_d
        dic[t] = self.rr
        return dic

    def new_state(self, cur_state, spd, rr):
        l_spd, r_spd = spd[0], spd[1]
        theta = cur_state[2]
        v = 0.5 * self.radius * (l_spd + r_spd)
        l_dis = rr * l_spd
        r_dis = rr * r_spd
        update = np.array((sy.cos(theta)*v*rr, sy.sin(theta)*v*rr, (r_dis-l_dis)/1.5))
        new_state = cur_state + update
        return new_state


    def res(self, msr):
        res = msr - self.H.dot(self.cur_state)
        res[2] = res[2] % (2*np.pi)
        if res[2] > np.pi:
            res[2] = res[2] - 2 * np.pi
        return res

    def update(self, res):
        K1 = np.dot(self.P, self.H.T)
        K2 = inv(np.dot(self.H, self.P).dot(self.H.T) + self.noise)
        self.K = K1.dot(K2)
        self.cur_state = self.cur_state + self.K.dot(res)
        self.P = (np.eye(3) - self.K.dot(self.H)).dot(self.P)

    def pred(self, speed):
        self.cur_state = self.new_state(self.cur_state, speed, self.rr)
        self.dic[self.l_spd] = speed[0]
        self.dic[self.r_spd] = speed[1]
        self.dic[self.theta] = self.cur_state[2]
        state_dev = np.array(self.state_dev.evalf(subs=self.dic)).astype(float)
        input_dev = np.array(self.input_dev.evalf(subs=self.dic)).astype(float)
        # diagonal matrix
        M = np.array([[self.motor_spd**2, 0], [0, self.motor_spd**2]])
        self.P = np.dot(state_dev, self.P).dot(state_dev.T) + np.dot(input_dev, M).dot(input_dev.T)
        self.t = self.t + 1

    def add_noise(self, cur_state):
        # rand = np.array([[random.uniform(0, 4), random.uniform(0, 4), random.uniform(0, 4)]])
        state_with_noise = cur_state + np.sqrt(self.noise).dot(np.random.randn(3))
        return state_with_noise

def Universe(robot, inc_time, stop_time):
    cur_state = robot.cur_state
    motor_spd = robot.motor_spd
    rr = robot.rr
    spd = np.array((np.pi/2, np.pi/2))
    length = 750
    width = 500
    overall_refresh = rr / inc_time
    x_points = []
    y_points = []
    x_old, y_old = [], []
    x_new, y_new = [], []
    x_msr, y_msr = [], []
    ctrl_steps = 20
    plt.figure(figsize=(20, 15))
    full_time = int(stop_time/inc_time)

    for i in range(full_time):
        real_spd = spd + motor_spd * np.random.randn(2)
        cur_state = robot.new_state(cur_state, real_spd, inc_time)
        cur_x = cur_state[0]
        cur_y = cur_state[1]
        x_points.append(cur_x)
        y_points.append(cur_y)



