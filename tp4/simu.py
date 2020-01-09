import pinocchio as pio
import numpy as np
import matplotlib.pylab as plt; plt.ion()
from numpy.linalg import inv, pinv, norm
pio.switchToNumpyMatrix()
from pinocchio.utils import rand,zero,rotate
from robot_hand import RobotHand
import time
import quadprog

np.set_printoptions(precision=5,linewidth=120,suppress=True) 
a2m = lambda a: np.matrix(a).T
m2a = lambda m: np.array(m.flat)

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


# index,col,res = getContactList(q)[0]    
# contact = res.getContact(0)

# #gv.addXYZaxis('world/ax',[1.,0.,0.,1.],1e-3,.03); gv.applyConfiguration('world/ax',[0,0,.05,0,0,0,1]);gv.refresh()
# robot.displayContact(contact,refresh=True)

def getContactJacobian(col,res):
    contact = res.getContact(0)
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

    a1 = rdata.a[joint1]
    a2 = rdata.a[joint2]
    a = (cMj1*a1-cMj2*a2).linear[2]
    
    return J,a

qdes = np.matrix([ 3.17,  0.12,  0.59,  0.87,  0.67, -2.1 , -1.8 , -1.8 , -0.4 , -0.28,  0.36,  0.48, -0.45,  0.7 ]).T
vqdes = 0*vq
vq *= 0
q[7]+=2

rmodel.gravity*=0

hq = []
hv = []
hd = []
hvd = []

previously = []
for it in range(10000):
    t=it*dt

    M = pio.crba(robot.model, robot.data, q)
    b = pio.rnea(robot.model, robot.data, q, vq, zero(robot.model.nv))
    pio.computeAllTerms(rmodel,rdata,q,vq)
    
    tauq = -Kp*(q-qdes) - Kv*(vq-vqdes)
    #tauq = zero(robot.model.nv)
    
    afree = inv(M)*(tauq-b)
    
    contacts = getContactList(q)
    pio.computeDistances(rmodel,rdata,gmodel,gdata,q)

      

    #contacts = [[0,gmodel.collisionPairs[0],gdata.collisionResults[0]]]
    if len(contacts)>0:
        for ic,[i,c,r] in enumerate(contacts):
            robot.displayContact(r.getContact(0),ic)
        J = np.vstack([ getContactJacobian(c,r)[0] for (i,c,r) in contacts ])
        dist = np.matrix([ gdata.distanceResults[i].min_distance for (i,c,r) in contacts ]).T

        for i,_,_ in contacts:
            if i not in previously:
                print('impact',it)
                vq -= pinv(J)*J*vq
                vq *=0
        vq -= pinv(J)*J*vq
        pio.forwardKinematics(rmodel,rdata,q,vq,0*vq)
        a0 = np.vstack([ getContactJacobian(c,r)[1] for (i,c,r) in contacts ])
            
        d = 0*-a0 #- 1000*dist - 1000*J*vq
        d = -1e3*J*vq
        hvd.append((J*vq).A[:,0])
        if it>=218: break

        '''
        min || a-afree ||_M^2/2 = (a-afree)M(a-afree)/2 = aMa/2 - afreeMa + afree**2
        s.t J a >= 0
        '''
        #aq,_,_,_,forces,ncontacts = quadprog.solve_qp(M,m2a(M*afree),-J.T,np.zeros(len(contacts)))
        aq,_,_,_,forces,ncontacts = quadprog.solve_qp(M,m2a(M*afree),-J.T,-m2a(d),meq=1)
        aq = a2m(aq)
    else:
        aq = afree

    previously = [ i for (i,c,r) in contacts ]

    hd.append([ gdata.distanceResults[0].min_distance ])

    vq += aq * dt
    q = pio.integrate(robot.model, q, vq * dt)

    if not it % 20:  # Only display once in a while ...
        robot.hideContact(-len(contacts))
        robot.display(q)
        time.sleep(1e-1)

    hq.append(q.copy().A[:,0])
    hv.append(vq.copy().A[:,0])


import collision_wrapper
col=collision_wrapper.CollisionWrapper(robot,gv)
col.computeCollisions(q)
cs=col.getCollisionList()
col.displayCollisions(cs)
Jt = col.computeCollisionJacobian(cs,q)
at0 = col.computeCollisionJdotQdot(cs,q,vq)
