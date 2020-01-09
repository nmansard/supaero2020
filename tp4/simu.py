import pinocchio as pio
import numpy as np
import matplotlib.pylab as plt; plt.ion()
from numpy.linalg import inv, pinv, norm
pio.switchToNumpyMatrix()
from pinocchio.utils import rand,zero,rotate
from robot_hand import RobotHand
import time

np.set_printoptions(precision=2,linewidth=120,suppress=True) 

robot = RobotHand()
gv = robot.viewer

rmodel = robot.model
rdata  = rmodel.createData()
gmodel = robot.gmodel
gdata  = gmodel.createData()
gdata.collisionRequest.enable_contact = True

Kp = 50.
Kv = 2 * np.sqrt(Kp)

q = np.matrix([ 3.17241553,  0.12073722,  0.59328528,  0.87136311,  0.66834531,
                -1.64030291, -1.92294792, -1.71647696, -0.39812831, -0.28055413,
                0.36374184,  0.48181465, -0.44956684,  0.70342902]).T
vq = rand(robot.model.nv)
robot.display(q)

dt = 1e-3

assert(pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False))

def getContactList(q):
    '''Return a list of triplets [ index,collision,result ] where index is the index of the 
    collision pair, colision is gmodel.collisionPairs[index] and result is gdata.collisionResults[index]. '''
    if pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False):
        return [ [ir,gmodel.collisionPairs[ir],r]
                 for ir,r in enumerate(gdata.collisionResults) if r.isCollision() ]
    else: return []


index,col,res = getContactList(q)[0]    
contact = res.getContact(0)

#gv.addXYZaxis('world/ax',[1.,0.,0.,1.],1e-3,.03); gv.applyConfiguration('world/ax',[0,0,.05,0,0,0,1]);gv.refresh()
robot.displayContact(contact,refresh=True)

def getContactJacobian(col,res):
    g1 = gmodel.geometryObjects[col.first]
    g2 = gmodel.geometryObjects[col.second]
    oMc = pio.SE3(pio.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,contact.normal).matrix(),contact.pos)

    joint1 = g1.parentJoint
    joint2 = g2.parentJoint
    oMj1 = rdata.oMi[joint1]
    oMj2 = rdata.oMi[joint2]

    cMj1 = oMc.inverse()*oMj1
    cMj2 = oMc.inverse()*oMj2

    J1=pio.computeJointJacobian(rmodel,rdata,q,joint1)
    J2=pio.computeJointJacobian(rmodel,rdata,q,joint2)
    Jc1=cMj1.action*J1
    Jc2=cMj2.action*J2
    J = (Jc1-Jc2)[2,:]

for it in range(1000):
    t=it*dt

    M = pio.crba(robot.model, robot.data, q)
    b = pio.rnea(robot.model, robot.data, q, vq, zero(robot.model.nv))
    tauq = zero(robot.model.nv)
    
    contacts = getContactList(q)
    if len(contacts)==0:
        aq = inv(M)*(tauq-b)
    else:
        J = np.vstack([ getContactJacobian(c,r) for (i,c,r) in contacts ])
        

        
        
