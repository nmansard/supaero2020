'''
Stand-alone program to display a 2-R robot at a random configuration q.
'''

import numpy as np
import gviewserver


def placement(x, y, theta):
    return [y, 0, x, 0, np.sin(theta / 2), 0, np.cos(theta / 2)]


gv = gviewserver.GepettoViewerServer()

gv.addSphere('world/joint1', .1, [1, 0, 0, 1])
gv.addSphere('world/joint2', .1, [1, 0, 0, 1])
gv.addSphere('world/joint3', .1, [1, 0, 0, 1])
gv.addCapsule('world/arm1', .05, .75, [1, 1, 1, 1])
gv.addCapsule('world/arm2', .05, .75, [1, 1, 1, 1])

q = np.matrix(np.random.rand(2) * 6 - 3).T

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
