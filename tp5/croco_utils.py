def plotDDPConvergence(costs, muLM, muV, gamma, theta, alpha, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figIndex, figsize=(6.4, 8))
    # Plotting the total cost sequence
    plt.subplot(511)
    plt.ylabel('cost')
    totalCost = [sum(cost) for cost in costs]
    plt.plot(totalCost)

    # Ploting mu sequences
    plt.subplot(512)
    plt.ylabel('mu')
    plt.plot(muLM, label='LM')
    plt.plot(muV, label='V')
    plt.legend()

    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(513)
    plt.ylabel('gamma')
    plt.plot(gamma)
    plt.subplot(514)
    plt.ylabel('theta')
    plt.plot(theta)

    # Plotting the alpha sequence
    plt.subplot(515)
    plt.ylabel('alpha')
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel('iteration')
    if show:
        plt.show()


def plotOCSolution(xs, us, figIndex=1, show=True):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx = xs[0].shape[0]
    nu = us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        U[i] = [u[i] for u in us]

    plt.figure(figIndex)

    # Plotting the state trajectories
    plt.subplot(211)
    [plt.plot(X[i], label='x' + str(i)) for i in range(nx)]
    plt.legend()

    # Plotting the control commands
    plt.subplot(212)
    [plt.plot(U[i], label='u' + str(i)) for i in range(nu)]
    plt.legend()
    plt.xlabel('knots')
    if show:
        plt.show()


def displayTrajectory(robot, xs, dt=0.1, rate=-1, cameraTF=None):
    """  Display a robot trajectory xs using Gepetto-viewer gui.

    :param robot: Robot wrapper
    :param xs: state trajectory
    :param dt: step duration
    :param rate: visualization rate
    :param cameraTF: camera transform
    """
    if not hasattr(robot, 'viewer'):
        robot.initDisplay(loadModel=True)
    if cameraTF is not None:
        robot.viewer.gui.setCameraTransform(0, cameraTF)
    import numpy as np

    import time
    S = 1 if rate <= 0 else max(len(xs) / rate, 1)
    for i, x in enumerate(xs):
        if not i % S:
            robot.display(x[:robot.nq])
            time.sleep(dt)
