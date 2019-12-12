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
IDX_GAZE = robot.model.getFrameId('framegaze')

gv.addXYZaxis('world/framegaze', [1., 0., 0., 1.], .03, .1)
gv.addSphere('world/ball',.1,[ 1.,0.,0.,1.])
ball = np.matrix([ 0.1,0.2,1.0 ]).T
gv.applyConfiguration('world/ball', list(ball.flat)+[1,0,0,0])
robot.display(robot.q0)

oMgoal = pio.SE3(pio.Quaternion(-0.4, 0.02, -0.5, 0.7).normalized().matrix(),
                np.matrix([.2, -.4, .7]).T)
gv.addXYZaxis('world/framegoal', [1., 0., 0., 1.], .015, .2)
gv.applyConfiguration('world/framegoal', list(pio.se3ToXYZQUAT(oMgoal).flat))

DT = 1e-2  # Integration step.
q0 = np.matrix([[ 0.  ,  0.  ,  0.  ,  1.  ,  0.18,  1.37, -0.24, -0.98,  0.98,
                  0.  ,  0.  ,  0.  ,  0.  , -0.13,  0.  ,  0.  ,  0.  ,  0.  ]]).T

q = q0.copy()
herr = [] # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(200):  # Integrate over 2 second of robot life
    pio.framesForwardKinematics(robot.model, robot.data, q)  # Compute frame placements
    oMgaze = robot.data.oMf[IDX_GAZE]           # Placement from world frame o to frame f oMgaze
    oRgaze = oMgaze.rotation                    # Rotation from world axes to gaze axes oRgaze 
    gaze_Jgaze = pio.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE)  # 6D jacobian in local frame
    o_Jgaze3 = oRgaze*gaze_Jgaze[:3,:]          # 3D jacobian in world frame
    o_GazeBall = oMgaze.translation-ball        # vector from gaze to ball, in world frame
    
    vq = -pinv(o_Jgaze3)*o_GazeBall

    q = pio.integrate(robot.model,q, vq * DT)
    robot.display(q)
    time.sleep(1e-3)

    herr.append(o_GazeBall) 

q = q0.copy()
herr = [] # Log the value of the error between tool and goal.
herr2 = [] # Log the value of the error between gaze and ball.
# Loop on an inverse kinematics for 200 iterations.
for i in range(200):  # Integrate over 2 second of robot life
    pio.framesForwardKinematics(robot.model, robot.data, q)  # Compute frame placements
    oMgaze = robot.data.oMf[IDX_GAZE]           # Placement from world frame o to frame f oMgaze
    oRgaze = oMgaze.rotation                    # Rotation from world axes to gaze axes oRgaze 
    gaze_Jgaze = pio.computeFrameJacobian(robot.model, robot.data, q, IDX_GAZE)  # 6D jacobian in local frame
    o_Jgaze3 = oRgaze*gaze_Jgaze[:3,:]          # 3D jacobian in world frame
    o_GazeBall = oMgaze.translation-ball        # vector from gaze to ball, in world frame
    
    pio.framesForwardKinematics(robot.model, robot.data, q)  # Compute frame placements
    oMtool = robot.data.oMf[IDX_TOOL]           # Placement from world frame o to frame f oMtool
    oRtool = oMtool.rotation                    # Rotation from world axes to tool axes oRtool 
    tool_Jtool = pio.computeFrameJacobian(robot.model, robot.data, q, IDX_TOOL)  # 6D jacobian in local frame
    o_Jtool3 = oRtool*tool_Jtool[:3,:]          # 3D jacobian in world frame
    o_TG = oMtool.translation-oMgoal.translation  # vector from tool to goal, in world frame

    vq = -pinv(o_Jtool3)*o_TG
    Ptool = eye(robot.nv)-pinv(o_Jtool3)*o_Jtool3
    vq += pinv(o_Jgaze3*Ptool)*(-o_GazeBall - o_Jgaze3*vq)

    q = pio.integrate(robot.model,q, vq * DT)
    robot.display(q)
    time.sleep(1e-3)

    herr.append(o_TG)
    herr2.append(o_GazeBall) 

