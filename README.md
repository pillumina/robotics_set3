# robotics_set3

# Preliminaries
## 0(a)
https://github.com/pillumina/robotics_set3
## 0(b)
Individually
## 0(c)
The resources in UKF_ref folder are unused in my solutions
## 0(d) & 0(e)
100% contribution from one person


# Mathematical Setup
## Define the system model
'''
    
    
    
    x_pos, y_pos, theta, l_spd, r_spd, radius, d, t = sy.symbols('x_pos, y_pos, theta, l_spd, r_spd, radius, d, t')
    self.func = self.symbolize(x_pos, y_pos, theta, l_spd, r_spd, radius, d, t)
    self.state_dev = self.func.jacobian(sy.Matrix([x_pos, y_pos, theta]))
    self.input_dev = self.func.jacobian(sy.Matrix([l_spd, r_spd]))
     
    def symbolize(self,x_pos, y_pos, theta, l_spd, r_spd, radius, d, t):
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

'''

## Extended Kalman Filter(EKF)

'''


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
     
     # input the measurement of state and return the residual.
     def res(self, msr):
        res = msr - self.H.dot(self.cur_state)
        res[2] = res[2] % (2*np.pi)
        if res[2] > np.pi:
            res[2] = res[2] - 2 * np.pi
        return res
'''

# Evaluation

![Change speeds of two wheels along teh time](EKF_traj.png)
