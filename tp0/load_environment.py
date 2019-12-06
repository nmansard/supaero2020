'''
Load a UR5 robot model, display it in the viewer.  Also create an obstacle
field made of several capsules, display them in the viewer and create the
collision detection to handle it.
'''

import pinocchio as pio
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
import eigenpy
import itertools
eigenpy.switchToNumpyMatrix()

exampleRobotDataPath = '/opt/openrobots/share/example-robot-data/robots/'
def createUR5WithObstacles(path = exampleRobotDataPath,
                           urdf = exampleRobotDataPath + 'ur_description/urdf/ur5_gripper.urdf',
                           initViewer = True):

    ### Robot
    # Load the robot
    robot = RobotWrapper.BuildFromURDF( urdf, [ path, ] )
    if initViewer: robot.initViewer( loadModel = True )

    ### Obstacle map
    # Capsule obstacles will be placed at these XYZ-RPY parameters
    oMobs = [ [ 0.40,  0.,  0.30, np.pi/2,0,0],
              [-0.08, -0.,  0.69, np.pi/2,0,0],
              [ 0.23, -0.,  0.04, np.pi/2, 0 ,0 ],
              [-0.32,  0., -0.08, np.pi/2, 0, 0]]

    # Load visual objects and add them in collision/visual models
    color = [ 1.0, 0.2, 0.2, 1.0 ]                       # color of the capsules
    rad,length = .1,0.2                                  # radius and length of capsules
    for i,xyzrpy in enumerate(oMobs):
        obs = pio.GeometryObject.CreateCapsule(rad,length)  # Pinocchio obstacle object
        obs.name = "obs%d"%i                                # Set object name
        obs.parentJoint = 0                                 # Set object parent = 0 = universe
        obs.placement = pio.SE3( rotate('x',xyzrpy[3]) * rotate('y',xyzrpy[4]) * rotate('z',xyzrpy[5]), 
                                 np.matrix([xyzrpy[:3]]).T )  # Set object placement wrt parent
        robot.collision_model.addGeometryObject(obs)  # Add object to collision model
        robot.visual_model   .addGeometryObject(obs)  # Add object to visual model
        # Also create a geometric object in gepetto viewer, with according name.
        robot.viewer.gui.addCapsule( "world/pinocchio/collisions/"+obs.name, rad,length,color )
        robot.viewer.gui.addCapsule( "world/pinocchio/visuals/"+obs.name, rad,length, [ 1.0, 0.2, 0.2, 1.0 ] )

    ### Collision pairs
    nobs = len(oMobs)
    nbodies = robot.collision_model.ngeoms-nobs
    robotBodies = range(nbodies)
    envBodies = range(nbodies,nbodies+nobs)
    for a,b in itertools.product(robotBodies,envBodies):
        robot.collision_model.addCollisionPair(pio.CollisionPair(a,b))
    
    ### Geom data
    # Collision/visual models have been modified => re-generate corresponding data.
    robot.collision_data = pio.GeometryData(robot.collision_model)
    robot.visual_data    = pio.GeometryData(robot.visual_model   )
    robot.viz.collision_model = robot.collision_model
    robot.viz.visual_model    = robot.visual_model
    robot.viz.collision_data = robot.collision_data
    robot.viz.visual_data    = robot.visual_data

    robot.viz.displayCollisions(False)

    return robot



class Target:
    '''
    Simple class target that stores and display the position of a target.
    '''
    def __init__(self,viewer,color = [ .0, 1.0, 0.2, 1.0 ], size = 0.05, position=None):
          self.name = "world/pinocchio/target"
          self.position = position if position is not None else np.matrix([ 0.0,  0.0 ]).T
          self.viewer = viewer
          self.viewer.gui.addCapsule( self.name, size,0., color)
          if position is not None and viewer is not None: self.display()
    def display(self):
         self.viewer.gui.applyConfiguration( self.name,
                                               [ self.position[0,0], 0, self.position[1,0],
                                                 1.,0.,0.0,0. ])
         self.viewer.gui.refresh()
