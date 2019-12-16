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

import hppfcl
gdata.collisionRequest.enable_contact=True
pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False)
r = gdata.collisionResults[2]
assert(r.isCollision())
c=r.getContact(0)
print(c.normal.T)

def Capsule(name,joint,placement,radius,length,gv=None):
    caps = pio.GeometryObject.CreateCapsule(radius,length)
    caps.name = name
    caps.placement = placement
    caps.parentJoint = joint
    if gv is not None:
        gv.addCapsule(name, radius,length, [1.,0.,0.,1.])
    return caps
from pinocchio.utils import rotate
#def addCapsule(self,name,joint,placement,r,l):

#gv.deleteNode('world/test1',True)
#gv.deleteNode('world/test2',True)
for n in gv.getNodeList(): gv.setVisibility(n,'OFF')

#robot.gmodel.addGeometryObject(Capsule('world/test1',0,pio.SE3(rotate('x',0.),np.matrix([0,0,0]).T),.1,1.,gv))
#robot.gmodel.addGeometryObject(Capsule('world/test2',0,pio.SE3(rotate('x',0.),np.matrix([0,0,0]).T),.1,1.,gv))



gv.addCapsule('world/wc190', .1,1., [1.,0.,0.,1.])
