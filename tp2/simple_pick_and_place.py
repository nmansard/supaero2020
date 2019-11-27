import math
import time
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
import pinocchio as pio
import eigenpy
eigenpy.switchToNumpyMatrix()

exampleRobotDataPath = '/opt/openrobots/share/example-robot-data/'
urdf = exampleRobotDataPath + 'ur_description/urdf/ur5_gripper.urdf'
robot = RobotWrapper.BuildFromURDF( urdf, [ exampleRobotDataPath, ] )
robot.initViewer(loadModel=True)
NQ = robot.model.nq
NV = robot.model.nv

# Add a red box
gv = robot.viewer.gui
boxID = "world/box"
rgbt = [1.0, 0.2, 0.2, 1.0]  # red-green-blue-transparency
gv.addBox(boxID, 0.05, 0.1, 0.1, rgbt)  # id, dim1, dim2, dim3, color
# Place the box at the position ( 0.5, 0.1, 0.2 )
q_box = [0.5, 0.1, 0.2, 1, 0, 0, 0]
gv.applyConfiguration(boxID, q_box)
gv.refresh()

#
# PICK #############################################################
#

# Configuration for picking the box
q = zero(NQ)
q[0] = -0.375
q[1] = -1.2
q[2] = 1.71
q[3] = -q[1] - q[2]
q[4] = q[0]

robot.display(q)
print("The robot is display with end effector on the red box.")
time.sleep(2)

#
# MOVE #############################################################
#

print("Let's start the movement ...")

# Random velocity of the robot driving the movement
vq = np.matrix([ 2.,0,0,4.,0,0]).T

idx = robot.index('wrist_3_joint')
oMeff = robot.placement(q, idx)  # Placement of end-eff wrt world at current configuration
oMbox = pio.XYZQUATToSe3(q_box)  # Placement of box     wrt world
effMbox = oMeff.inverse() * oMbox  # Placement of box     wrt eff

for i in range(1000):
    # Chose new configuration of the robot
    q += vq / 40
    q[2] = 1.71 + math.sin(i * 0.05) / 2

    # Gets the new position of the box
    oMbox = robot.placement(q, idx) * effMbox

    # Display new configuration for robot and box
    gv.applyConfiguration(boxID, pio.se3ToXYZQUATtuple(oMbox))
    robot.display(q)
    time.sleep(0.1)
