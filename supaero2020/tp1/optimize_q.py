'''
Stand-alone program to optimize the configuration q=[q1,q2] of a 2-R robot with
scipy BFGS.
'''

import time

import numpy as np
from scipy.optimize import fmin_bfgs

import gviewserver


def placement(x, y, theta):
    return [y, 0, x, 0, np.sin(theta / 2), 0, np.cos(theta / 2)]


gv = gviewserver.GepettoViewerServer()

gv.addSphere('world/joint1', .1, [1, 0, 0, 1])
gv.addSphere('world/joint2', .1, [1, 0, 0, 1])
gv.addSphere('world/joint3', .1, [1, 0, 0, 1])
gv.addCapsule('world/arm1', .05, .75, [1, 1, 1, 1])
gv.addCapsule('world/arm2', .05, .75, [1, 1, 1, 1])


def display2d(q):
    '''Display the robot in Gepetto Viewer. '''
    assert (q.shape == (2, 1))
    c0 = np.cos(q[0, 0])
    s0 = np.sin(q[0, 0])
    c1 = np.cos(q[0, 0] + q[1, 0])
    s1 = np.sin(q[0, 0] + q[1, 0])
    gv.applyConfiguration('world/joint1', placement(0, 0, 0))
    gv.applyConfiguration('world/arm1', placement(c0 / 2, s0 / 2, q[0, 0]))
    gv.applyConfiguration('world/joint2', placement(c0, s0, q[0, 0]))
    gv.applyConfiguration('world/arm2', placement(c0 + c1 / 2, s0 + s1 / 2, q[0, 0] + q[1, 0]))
    gv.applyConfiguration('world/joint3', placement(c0 + c1, s0 + s1, q[0, 0] + q[1, 0]))
    gv.refresh()


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
    time.sleep(.5)


x0 = np.array([0.0, 0.0])
xopt_bfgs = fmin_bfgs(cost, x0, callback=callback)
print('\n *** Xopt in BFGS = %s \n\n\n\n' % xopt_bfgs)
