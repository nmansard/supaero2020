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

colwrap = CollisionWrapper(robot)
rmodel,rdata = colwrap.rmodel,colwrap.rdata

Kp = 50.
Kv = 2 * np.sqrt(Kp)
dt = 1e-3

q = robot.q0.copy()
vq = zero(robot.model.nv)

qdes = np.matrix([ 3.17,  0.12,  0.59,  0.87,  0.67, -2.1 , -1.8 , -1.8 , -0.4 , -0.28,  0.36,  0.48, -0.45,  0.7 ]).T
vqdes = 0*vq

hq = []
hv = []
hd = []
hvd = []

previously = []
for it in range(10000):
    t=it*dt

    tauq = -Kp*(q-qdes) - Kv*(vq-vqdes)

    pio.computeAllTerms(rmodel,rdata,q,vq)
    M = rdata.M
    b = rdata.nle
    afree = inv(M)*(tauq-b)

    colwrap.computeCollisions(q)
    collisions = colwrap.getCollisionList()
      
    if len(collisions)>0:
        J = colwrap.getCollisionJacobian(collisions)
        dist = colwrap.getCollisionDistances(collisions)
        
        for i,_,_ in collisions:
            if i not in previously:
                print('impact',it)
                vq -= pinv(J)*J*vq

        a0 = colwrap.getCollisionJdotQdot(collisions)
        d = -a0 - 1000*dist - 1000*J*vq
        
        hd.append(dist)
        hvd.append((J*vq).A[:,0])
        
        '''
        min || a-afree ||_M^2/2 = (a-afree)M(a-afree)/2 = aMa/2 - afreeMa + afree**2
        s.t J a >= 0
        '''
        aq,_,_,_,forces,ncontacts = quadprog.solve_qp(M,m2a(M*afree),J.T,m2a(d))
        aq = a2m(aq)
    else:
        aq = afree

    previously = [ i for (i,c,r) in collisions ]

    vq += aq * dt
    q = pio.integrate(robot.model, q, vq * dt)

    if not it % 20:  # Only display once in a while ...
        colwrap.displayCollisions(collisions)
        robot.display(q)
        time.sleep(1e-1)

    hq.append(q.copy().A[:,0])
    hv.append(vq.copy().A[:,0])

