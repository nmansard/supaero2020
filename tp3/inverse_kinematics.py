'''
Inverse kinematics (close loop / iterative) for a mobile manipulator.
Template of the program for TP3
'''

import pinocchio as pio
from pinocchio.utils import *
import time
from numpy.linalg import pinv
from tiago_loader import loadTiago
pio.switchToNumpyMatrix()
import matplotlib.pylab as plt
plt.ion()

robot = loadTiago(initViewer=True)

gv = robot.viewer.gui
gv.setCameraTransform(0, [-8, -8, 2, .6, -0.25, -0.25, .7])

NQ = robot.model.nq
NV = robot.model.nv
IDX_TOOL = robot.model.getFrameId('frametool')
IDX_BASIS = robot.model.getFrameId('framebasis')

oMgoal = pio.SE3(pio.Quaternion(-0.4, 0.02, -0.5, 0.7).normalized().matrix(),
                np.matrix([.2, -.4, .7]).T)
gv.addXYZaxis('world/framegoal', [1., 0., 0., 1.], .015, .2)
gv.applyConfiguration('world/framegoal', list(pio.se3ToXYZQUAT(oMgoal).flat))

DT = 1e-2  # Integration step.
q0 = np.matrix([[ 0.  ,  0.  ,  0.  ,  1.  ,  0.18,  1.37, -0.24, -0.98,  0.98,
                  0.  ,  0.  ,  0.  ,  0.  , -0.13,  0.  ,  0.  ,  0.  ,  0.  ]]).T

q = q0.copy()
herr = [] # Log the value of the error between tool and goal.
# Loop on an inverse kinematics for 200 iterations.
for i in range(200):  # Integrate over 2 second of robot life
    pio.framesForwardKinematics(robot.model, robot.data, q)  # Compute frame placements
    oMtool = robot.data.oMf[IDX_TOOL]           # Placement from world frame o to frame f oMtool
    oRtool = oMtool.rotation                    # Rotation from world axes to tool axes oRtool 
    tool_Jtool = pio.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)  # 6D jacobian in local frame
    o_Jtool3 = oRtool*tool_Jtool[:3,:]          # 3D jacobian in world frame
    o_TG = oMtool.translation-oMgoal.translation  # vector from tool to goal, in world frame
    
    vq = -pinv(o_Jtool3)*o_TG

    q = pio.integrate(robot.model,q, vq * DT)
    robot.display(q)
    time.sleep(1e-3)

    herr.append(o_TG) 

q = q0.copy()
herr = []
for i in range(1000):  # Integrate over 2 second of robot life
    pio.framesForwardKinematics(robot.model, robot.data, q)  # Compute frame placements
    oMtool = robot.data.oMf[IDX_TOOL]                 # Placement from world frame o to frame f oMtool  
    tool_nu = pio.log(oMtool.inverse()*oMgoal).vector  # 6D error between the two frame
    tool_Jtool = pio.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)  # Get corresponding jacobian
    vq = pinv(tool_Jtool)*tool_nu

    q = pio.integrate(robot.model,q, vq * DT)
    robot.display(q)
    time.sleep(1e-3)

    herr.append(tool_nu)

