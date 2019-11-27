'''
Stand-alone inverse geom in 3D.  Given a reference translation <target> ,
it computes the configuration of the UR5 so that the end-effector position (3D)
matches the target. This is done using BFGS solver. While iterating to compute
the optimal configuration, the script also display the successive candidate
solution, hence moving the robot from the initial guess configuaration to the
reference target.
'''

import time
import numpy as np
import pinocchio as pio
from pinocchio.robot_wrapper import RobotWrapper
from scipy.optimize import fmin_bfgs
import eigenpy
eigenpy.switchToNumpyMatrix()

# --- Load robot model
exampleRobotDataPath = '/opt/openrobots/share/example-robot-data/'
urdf = exampleRobotDataPath + 'ur_description/urdf/ur5_gripper.urdf'
robot = RobotWrapper.BuildFromURDF( urdf, [ exampleRobotDataPath, ] )
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv

# Define an init config
robot.q0 = np.matrix([0, -3.14/2, 0, 0, 0, 0]).T
robot.display(robot.q0)
time.sleep(1)
print("Let's go to pdes.")

# --- Add ball to represent target
gv=robot.viewer.gui
gv.addSphere("world/ball", .05, [1., .2, .2, .5])
gv.addSphere("world/blue", .05, [.2, .2, 1., .5])

#
# OPTIM 3D #########################################################
#

def cost(q):
    '''Compute score from a configuration'''
    q = np.matrix(q).T
    p = robot.placement(q, 6).translation
    return np.linalg.norm(p - target)**2

def callback(q):
    q = np.matrix(q).T
    gv.applyConfiguration('world/ball', tuple(target.flat)+(0,1,0,0))
    gv.applyConfiguration('world/blue', pio.se3ToXYZQUATtuple(robot.placement(q, 6)))
    robot.display(q)
    time.sleep(1e-2)

target = np.matrix([0.5, 0.1, 0.2]).T  # x,y,z
qopt = fmin_bfgs(cost, robot.q0, callback=callback)
