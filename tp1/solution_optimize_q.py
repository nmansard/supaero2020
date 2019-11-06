import numpy as np
from scipy.optimize import fmin_bfgs


def placement(x, y, theta):
    return [y, 0, x, 0, np.sin(theta / 2), 0, np.cos(theta / 2)]


def display2d(q):
    '''Display the robot in Gepetto Viewer. '''
    assert (q.shape == (2, 1))
    c0 = np.cos(q[0, 0])
    s0 = np.sin(q[0, 0])
    c1 = np.cos(q[0, 0] + q[1, 0])
    s1 = np.sin(q[0, 0] + q[1, 0])
    gv.applyConfiguration('world/joint1', placement(0, 0, 0))  # noqa
    gv.applyConfiguration('world/arm1', placement(c0 / 2, s0 / 2, q[0, 0]))  # noqa
    gv.applyConfiguration('world/joint2', placement(c0, s0, q[0, 0]))  # noqa
    gv.applyConfiguration('world/arm2', placement(c0 + c1 / 2, s0 + s1 / 2, q[0, 0] + q[1, 0]))  # noqa
    gv.applyConfiguration('world/joint3', placement(c0 + c1, s0 + s1, q[0, 0] + q[1, 0]))  # noqa
    gv.refresh()  # noqa


def endeffector(q):
    '''Return the 2D position of the end effector of the robot at configuration q. '''
    assert (q.shape == (2, 1))
    c0 = np.cos(q[0, 0])
    s0 = np.sin(q[0, 0])
    c1 = np.cos(q[0, 0] + q[1, 0])
    s1 = np.sin(q[0, 0] + q[1, 0])
    return np.matrix([c0 + c1, s0 + s1]).T


target = np.matrix([.5, .5]).T


def cost(q):
    q = np.matrix(q).T
    eff = endeffector(q)
    return np.linalg.norm(eff - target)


def callback(q):
    q = np.matrix(q).T
    display2d(q)
    import time
    time.sleep(.1)


x0 = np.array([0.0, 0.0])
xopt_bfgs = fmin_bfgs(cost, x0, callback=callback)
print('\n *** Xopt in BFGS = %s \n\n\n\n' % xopt_bfgs)
