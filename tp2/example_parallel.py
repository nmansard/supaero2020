'''
Load 4 times the UR5 model, plus a plate object on top of them, to feature a simple parallel robot.
No optimization, this file is just an example of how to load the models.
'''

import time
import numpy as np
import pinocchio as pio
from example_robot_data import loadUR
from pinocchio.utils import *
from scipy.optimize import fmin_bfgs
import eigenpy
eigenpy.switchToNumpyMatrix()

def loadRobot(M0, name):
    '''
    This function load a UR5 robot n a new model, move the basis to placement <M0>
    and add the corresponding visuals in gepetto viewer with name prefix given by string <name>.
    It returns the robot wrapper (model,data).
    '''
    robot = loadUR()
    robot.model.jointPlacements[1] = M0 * robot.model.jointPlacements[1]
    robot.visual_model.geometryObjects[0].placement = M0 * robot.visual_model.geometryObjects[0].placement
    robot.visual_data.oMg[0] = M0 * robot.visual_data.oMg[0]
    robot.initViewer(loadModel=True, sceneName="world/" + name)
    return robot

robots = []
# Load 4 Ur5 robots, placed at 0.3m from origin in the 4 directions x,y,-x,-y.
Mt = pio.SE3(eye(3), np.matrix([.3, 0, 0]).T)  # First robot is simply translated
for i in range(4):
    basePlacement = pio.SE3(rotate('z', np.pi / 2 * i), zero(3)) * Mt
    robots.append(loadRobot(basePlacement, "robot%d" % i))
gv = robots[0].viewer.gui # any robots could be use to get gv.

# Set up the robots configuration with end effector pointed upward.
q0 = np.matrix([np.pi / 4, -np.pi / 4, -np.pi / 2, np.pi / 4, np.pi / 2, 0]).T
for i in range(4):
    robots[i].display(q0)

# Add a new object featuring the parallel robot tool plate.
[w, h, d] = [0.5, 0.5, 0.005]
color = [red, green, blue, transparency] = [1, 1, 0.78, .3]
gv.addBox('world/robot0/toolplate', w, h, d, color)
Mtool = pio.SE3(rotate('z', 1.268), np.matrix([0, 0, .75]).T)
gv.applyConfiguration('world/robot0/toolplate', pio.se3ToXYZQUATtuple(Mtool))
gv.refresh()
