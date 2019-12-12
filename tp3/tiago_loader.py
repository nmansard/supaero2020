import pinocchio as pio
from example_robot_data.robots_loader import getModelPath
import numpy
pio.switchToNumpyMatrix()

def loadTiago(initViewer=True):
    '''Load the Tiago robot, and add two operation frames:
    - at the front of the basis, named framebasis.
    - at the end effector, name frametool.
    Take care: Tiago as 3 continuous joints (one for the basis, two for the wheels), which makes 
    nq == nv+3.
    '''
    URDF_FILENAME = "tiago_no_hand.urdf"
    URDF_SUBPATH = "/tiago_description/robots/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)
    robot = pio.RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pio.JointModelPlanar())
    robot.model.addFrame(pio.Frame('framebasis',1,1,
                                   pio.SE3(numpy.eye(3),numpy.matrix([0.28,0,0.1]).T),
                                   pio.FrameType.OP_FRAME))
    robot.model.addFrame(pio.Frame('frametool',9,51,
                                   pio.SE3(numpy.matrix([[0,1.,0],[0,0,1],[1,0,0]]),
                                           numpy.matrix([0.,0,0.06]).T),
                                   pio.FrameType.OP_FRAME))
    robot.model.addFrame(pio.Frame('framegaze',11,57,
                                   pio.SE3(numpy.eye(3),
                                           numpy.matrix([0.4,0,0]).T),
                                   pio.FrameType.OP_FRAME))
    robot.data = robot.model.createData()

    jllow = robot.model.lowerPositionLimit
    jllow[:2] = -2
    robot.model.lowerPositionLimit = jllow
    jlupp = robot.model.upperPositionLimit
    jlupp[:2] = 2
    robot.model.upperPositionLimit = jlupp
    
    def display(robot,q):
        robot.realdisplay(q)
        pio.updateFramePlacements(robot.model,robot.viz.data)
        robot.viewer.gui.applyConfiguration('world/'+robot.model.frames[-2].name,
                                            list(pio.se3ToXYZQUAT(robot.viz.data.oMf[-2]).flat))
        robot.viewer.gui.applyConfiguration('world/'+robot.model.frames[-1].name,
                                            list(pio.se3ToXYZQUAT(robot.viz.data.oMf[-1]).flat))
        robot.viewer.gui.refresh()
        
    if initViewer:
        robot.initViewer(loadModel=True)
        gv = robot.viewer.gui
        gv.addFloor('world/floor')
        gv.addXYZaxis('world/framebasis', [1., 0., 0., 1.], .03, .1)
        gv.addXYZaxis('world/frametool', [1., 0., 0., 1.], .03, .1)
        #gv.addXYZaxis('world/framegaze', [1., 0., 0., 1.], .03, .1)

        robot.realdisplay = robot.display
        import types
        robot.display = types.MethodType(display,robot)

    return robot

