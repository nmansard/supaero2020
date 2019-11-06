# flake8: noqa


def penalty(x):
    return cost_9(x) + 100 * np.linalg.norm(constraint_9(x))**2


xopt_bfgs = fmin_bfgs(penalty, x0, callback=callback_9)
print('\n *** Xopt BFGS = %s \n\n\n\n' % xopt_bfgs)
