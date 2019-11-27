import time

import numpy as np
from pinocchio.utils import rand

p0 = rand(3)
p1 = rand(3)

for t in np.arange(0, 1, .01):
    p = p0 * (1 - t) + p1 * t
    gv.applyConfiguration('world/box', p.T.tolist()[0] + quat.coeffs().T.tolist()[0])  # noqa
    gv.refresh()  # noqa
    time.sleep(.01)
