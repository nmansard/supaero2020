import pinocchio as pio
import numpy as np
import matplotlib.pylab as plt; plt.ion()
from numpy.linalg import inv, pinv, norm
pio.switchToNumpyMatrix()
from pinocchio.utils import rand,zero,rotate
from robot_hand import RobotHand
import time
from traj_ref import TrajRef

robot = RobotHand()
gv = robot.viewer

q = robot.q0.copy()
vq = zero(robot.model.nv)

Kp = 50.
Kv = 2 * np.sqrt(Kp)
dt = 1e-3

qdes = TrajRef(robot.q0,omega = np.array([0,.1,1,1.5,2.5,-1,-1.5,-2.5,.1,.2,.3,.4,.5,.6]),amplitude=1.5)

hq = []
hqdes = []
for i in range(10000):

    M = pio.crba(robot.model, robot.data, q)
    b = pio.rnea(robot.model, robot.data, q, vq, zero(robot.model.nv))

    tauq = -Kp*(q-qdes(i*dt)) - Kv*(vq-qdes.velocity(i*dt)) + qdes.acceleration(i*dt)

    aq = inv(M) * (tauq - b)
    vq += aq * dt
    q = pio.integrate(robot.model, q, vq * dt)

    if not i % 3:  # Only display once in a while ...
        robot.display(q)
        time.sleep(1e-4)

    hq.append(q.copy().A[:,0])
    hqdes.append(qdes.copy().A[:,0])
        
