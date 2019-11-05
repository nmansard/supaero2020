import numpy as np


def placement(x, y, theta):
    return [y, 0, x, 0, np.sin(theta / 2), 0, np.cos(theta / 2)]


def solution_display_2r(q, gv):
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
