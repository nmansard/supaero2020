import pinocchio as pio
import numpy as np
import matplotlib.pylab as plt; plt.ion()
from numpy.linalg import inv, pinv, norm
pio.switchToNumpyMatrix()
from pinocchio.utils import rand,zero,rotate
from robot_hand import RobotHand
import time
import quadprog
from collision_wrapper import CollisionWrapper

np.set_printoptions(precision=5,linewidth=120,suppress=True) 
a2m = lambda a: np.matrix(a).T
m2a = lambda m: np.array(m.flat)

robot = RobotHand()
gv = robot.viewer

colwrap = CollisionWrapper(robot,gv)
rmodel,rdata = colwrap.rmodel,colwrap.rdata

Kp = 50.
Kv = 2 * np.sqrt(Kp)

q = np.matrix([ 3.17241553,  0.12073722,  0.59328528,  0.87136311,  0.66834531,
                -1.64030291, -1.92294792, -1.71647696, -0.39812831, -0.28055413,
                0.36374184,  0.48181465, -0.44956684,  0.70342902]).T
vq = rand(robot.model.nv)
robot.display(q)

dt = 1e-3

qdes = np.matrix([ 3.17,  0.12,  0.59,  0.87,  0.67, -2.1 , -1.8 , -1.8 , -0.4 , -0.28,  0.36,  0.48, -0.45,  0.7 ]).T
vqdes = 0*vq
vq *= 0
q[7]+=2

robot.model.gravity*=0

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

    colwrap.computeCollisions(q)
    contacts = colwrap.getCollisionList()
      
    if len(contacts)>0:
        J = colwrap.getCollisionJacobian(contacts)
        dist = colwrap.getCollisionDistances(contacts)
        
        for i,_,_ in contacts:
            if i not in previously:
                print('impact',it)
                vq -= pinv(J)*J*vq
                vq *=0
        vq -= pinv(J)*J*vq

        a0 = colwrap.getCollisionJdotQdot(contacts)
            
        d = 0*-a0 #- 1000*dist - 1000*J*vq
        d = -1e3*J*vq

        hd.append(dist)
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


    vq += aq * dt
    q = pio.integrate(robot.model, q, vq * dt)

    if not it % 20:  # Only display once in a while ...
        colwrap.displayCollisions(contacts)
        robot.display(q)
        time.sleep(1e-1)

    hq.append(q.copy().A[:,0])
    hv.append(vq.copy().A[:,0])

