"""Defines a differential drive robot
and makes it describe a simple path"""

from matplotlib import pyplot as plt
from math import sin, cos

class Robot(object):
    """Defines basic mobile robot properties"""
    def __init__(self):
        self.pos_x  = 0.0
        self.pos_y  = 0.0
        self.angle  = 0.0
        self.plot   = False
        self._delta = 0.01

    # Movement
    def step(self):
        """ updates the x,y and angle """
        self.deltax()
        self.deltay()
        self.deltaa()

    def move(self, seconds):
        """ Moves the robot for an 's' amount of seconds"""
        for i in range(int(seconds/self._delta)):
            self.step()
            if i % 3 == 0 and self.plot: # plot path every 3 steps
                self.plot_xya()

    # Printing-and-plotting:
    def print_xya(self):
        """ prints the x,y position and angle """
        print ("x = " + str(self.pos_x) +" "+ "y = " + str(self.pos_y))
        print ("a = " + str(self.angle))

    def plot_robot(self):
        """ plots a representation of the robot """
        plt.arrow(self.pos_x, self.pos_y, 0.001
                  * cos(self.angle), 0.001 * sin(self.angle),
                  head_width=self.length, head_length=self.length,
                  fc='k', ec='k')

    def plot_xya(self):
        """ plots a dot in the position of the robot """
        plt.scatter(self.pos_x, self.pos_y, c='r', edgecolors='r')


class DDRobot(Robot):
    """Defines a differential drive robot"""

    def __init__(self):
        Robot.__init__(self)
        self.radius = 0.1
        self.length = 0.4

        self.rt_spd_left = 0.0
        self.rt_spd_right = 0.0

    def deltax(self):
        """ update x depending on l and r angular speeds """
        self.pos_x += self._delta * (self.radius*0.5) \
        * (self.rt_spd_right + self.rt_spd_left)*cos(self.angle)

    def deltay(self):
        """ update y depending on l and r angular speeds """
        self.pos_y += self._delta * (self.radius*0.5) \
        * (self.rt_spd_right + self.rt_spd_left)*sin(self.angle)

    def deltaa(self):
        """ update z depending on l and r angular speeds """
        self.angle += self._delta * (self.radius/self.length) \
        * (self.rt_spd_right - self.rt_spd_left)


enesbot = DDRobot()            # robot called 'enesbot'

enesbot.angle = 3.1416/4        # 45 degrees
enesbot.plot = True             # plot the robot!
enesbot.plot_robot()

enesbot.rt_spd_left = 10
enesbot.rt_spd_right = 10       # straight line
enesbot.move(2)                 # move for 2 seconds

enesbot.rt_spd_left = 12.5664
enesbot.rt_spd_right = 18.8496  # (2m diameter circle)
enesbot.move(1)                 # move for 1 second

enesbot.rt_spd_left = 18.8496
enesbot.rt_spd_right = 12.5664  # (2m diameter circle)
enesbot.move(2.5)               # move for 2.5 second

enesbot.rt_spd_left = 12.5664
enesbot.rt_spd_right = 18.8496  # (2m diameter circle)
enesbot.move(2.5)               # move for 2.5 second

enesbot.plot_robot()

plt.xlim([-1, 6])               # axis limits
plt.ylim([-1, 6])

plt.show()