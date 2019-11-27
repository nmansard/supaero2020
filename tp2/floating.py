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
from example_robot_data import loadANYmal
from pinocchio.utils import *
from scipy.optimize import fmin_bfgs
import eigenpy
eigenpy.switchToNumpyMatrix()

# --- Load robot model
robot = loadANYmal()
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv
robot.display(robot.q0)

robot.feetIndexes = [ robot.model.getFrameId(frameName) for frameName in ['RH_FOOT','LH_FOOT','RF_FOOT','LF_FOOT' ] ]

# --- Add box to represent target
gv=robot.viewer.gui
gv.addSphere("world/red", .05, [1., .2, .2, .5])  # .1 is the radius
gv.addSphere("world/blue", .05, [.2, .2, 1., .5])  # .1 is the radius
gv.addSphere("world/green", .05, [.1, 1., .1, .5])  # .1 is the radius
gv.addSphere("world/purple", .05, [1., .2, 1., .5])  # .1 is the radius
#gv.addSphere("world/white", .05, [.9, .9, .9, .5])  # .1 is the radius

gv.addSphere("world/red_des", .05, [1., .2, .2, .95])  # .1 is the radius
gv.addSphere("world/blue_des", .05, [.2, .2, 1., .95])  # .1 is the radius
gv.addSphere("world/green_des", .05, [.1, 1., .1, .95])  # .1 is the radius
gv.addSphere("world/purple_des", .05, [1., .2, 1., .95])  # .1 is the radius
#gv.addSphere("world/white_des", .05, [.9, .9, .9, .95])  # .1 is the radius

#
# OPTIM 6D #########################################################
#

pdes = [
    np.matrix(  [[-0.7, -0.2, 1.2]]).T,
    np.matrix(  [[-0.3,  0.5,  0.8]]).T,
    np.matrix(  [[0.3, 0.1, -0.1]]).T,
    np.matrix(  [[0.9, 0.9, 0.5]]).T
    ]
for i in range(4): pdes[i][2]+=1
colors = ['red','blue','green','purple']

def cost(q):
    '''Compute score from a configuration'''
    q = np.matrix(q).T
    cost = 0.
    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i]).translation
        cost += np.linalg.norm(p_i-pdes[i])**2
    return cost

def callback(q):
    q = np.matrix(q).T
    gv.applyConfiguration('world/box', pio.se3ToXYZQUATtuple(Mtarget))

    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i])
        gv.applyConfiguration('world/'+colors[i], pio.se3ToXYZQUATtuple(p_i))
        gv.applyConfiguration('world/%s_des'%colors[i], tuple(pdes[i].flat)+(0,0,0,1))

    robot.display(q)
    time.sleep(1e-1)

Mtarget = pio.SE3(rotate('x',3.14/4), np.matrix([0.5, 0.1, 0.2]).T)  # x,y,z
qopt = fmin_bfgs(cost, robot.q0, callback=callback)
qopt = np.matrix(qopt).T

