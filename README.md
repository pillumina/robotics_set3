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

## Extended Kalman Filter(EKF)
'''python
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

'''

# Evaluation
