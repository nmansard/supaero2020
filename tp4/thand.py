import pinocchio as pio
pio.switchToNumpyMatrix()
from pinocchio.utils import *
from robot_hand_fcl import Robot
import time

robot = Robot()
robot.display(robot.q0)

q = robot.q0.copy()
robot.display(q)

rmodel = robot.model
rdata  = rmodel.createData()
gmodel = robot.gmodel
gdata  = gmodel.createData()
gdata.collisionRequest.enable_contact = True

gv = robot.viewer
#gv.setCameraTransform(0,[-0.14, 0.25, 0.22, -0.13, 0.43, 0.81, -0.38])

q[2:5].flat = [0,0,0]
q[0]=-np.pi/2
pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False)
robot.display(q)

#q[2:5] = rand(3)*3-1
q.flat[2:5] = [-.3,-.2,-.1]
vq = zero(rmodel.nv)
#vq[2:5] = rand(3)*2-1
vq.flat[2:5] = [-1,-1,-0]

for i in range(1000):
    #q[2:5] = rand(3)*3-1
    q += vq*1e-2
    robot.display(q)
    
    if pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False):
        break

pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False)
r = gdata.collisionResults[2]
assert(r.isCollision())
c=r.getContact(0)
print(c.normal.T)

