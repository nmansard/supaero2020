'''
Stand-alone program to optimize the placement of a 2d robot, where the decision variables
are the placement of the 3 bodies of the robot. BFGS and SLSQP solvers are used.
'''

import time

import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp

import gviewserver

gv = gviewserver.GepettoViewerServer()


def placement(x, y, theta):
    return [y, 0, x, 0, np.sin(theta / 2), 0, np.cos(theta / 2)]


cos = np.cos
sin = np.sin
norm = np.linalg.norm

gv.addSphere('world/joint1', .1, [1, 0, 0, 1])
gv.addSphere('world/joint2', .1, [1, 0, 0, 1])
gv.addSphere('world/joint3', .1, [1, 0, 0, 1])
gv.addCapsule('world/arm1', .05, .75, [1, 1, 1, 1])
gv.addCapsule('world/arm2', .05, .75, [1, 1, 1, 1])


def display(ps):
    '''Display the robot in Gepetto Viewer. '''
    assert (ps.shape == (9, ))

    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps

    gv.applyConfiguration('world/joint1', placement(x1, y1, t1))
    gv.applyConfiguration('world/arm1', placement(x1 + np.cos(t1) / 2, x1 + np.sin(t1) / 2, t1))

    gv.applyConfiguration('world/joint2', placement(x2, y2, t2))
    gv.applyConfiguration('world/arm2', placement(x2 + np.cos(t2) / 2, y2 + np.sin(t2) / 2, t2))

    gv.applyConfiguration('world/joint3', placement(x3, y3, t3))

    gv.refresh()


def endeffector(ps):
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    return np.matrix([x3, y3]).T


target = np.matrix([.5, .5]).T


def cost(ps):
    eff = endeffector(ps)
    return norm(eff - target)**2


def constraint(ps):
    assert (ps.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = ps
    res = np.zeros(6)
    res[0] = x1 - 0
    res[1] = y1 - 0
    res[2] = x1 + cos(t1) - x2
    res[3] = y1 + sin(t1) - y2
    res[4] = x2 + cos(t2) - x3
    res[5] = y2 + sin(t2) - y3
    return res


def penalty(ps):
    return cost(ps) + 10 * sum(np.square(constraint(ps)))


def callback(ps):
    display(ps)
    time.sleep(.5)


x0 = np.array([
    0.0,
] * 9)

with_bfgs = 0
if with_bfgs:
    xopt = fmin_bfgs(penalty, x0, callback=callback)
else:
    xopt = fmin_slsqp(cost, x0, callback=callback, f_eqcons=constraint, iprint=2, full_output=1)[0]

print('\n *** Xopt = %s\n\n\n\n' % xopt)
