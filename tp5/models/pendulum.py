'''
Create a simulation environment for a N-pendulum.
Example of use:

env = Pendulum(N)
env.reset()

for i in range(1000):
   env.step(zero(env.nu))
   env.render()

'''

from pinocchio.utils import *
import pinocchio as pio
from gviewserver import GepettoViewerServer
pio.switchToNumpyArray()

# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------

def Capsule(name,joint,placement,radius,length):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    caps = pio.GeometryObject.CreateCapsule(radius,length)
    caps.name = name
    caps.placement = placement
    caps.parentJoint = joint
    return caps

def getViewerPrefix(viewer,path):
    if viewer is None: return ""
    viewerPrefix = ""
    for s in path:
        viewerPrefix += s
        viewer.createGroup(viewerPrefix)
        viewerPrefix += "/"
    return viewerPrefix

def createPendulum(nbJoint,length=1.0,mass=1.0,viewer=None):
    '''
    Creates the Pinocchio kinematic <rmodel> and visuals <gmodel> models for 
    a N-pendulum.

    @param nbJoint: number of joints <N> of the N-pendulum.
    @param length: length of each arm of the pendulum.
    @param mass: mass of each arm of the pendulum.
    @param viewer: gepetto-viewer CORBA client. If not None, then creates the geometries
    in the viewer.
    '''
    rmodel = pio.Model()
    gmodel = pio.GeometryModel()
    
    color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
    colorred = [1.0,0.0,0.0,1.0]

    radius = 0.1*length
    
    prefix = ''
    jointId = 0
    jointPlacement = pio.SE3.Identity()
    inertia = pio.Inertia(mass,
                          np.matrix([0.0,0.0,length/2]).T,
                          mass/5*np.diagflat([ 1e-2,length**2,  1e-2 ]) )
    viewerPrefix = getViewerPrefix(viewer,["world","pinocchio","visuals"])

    for i in range(nbJoint):
        istr = str(i)
        name               = prefix+"joint"+istr
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = rmodel.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        rmodel.appendBodyToJoint(jointId,inertia,pio.SE3.Identity())

        if viewer is not None:
            viewer.addSphere(viewerPrefix+jointName, 1.5*radius,colorred)
            viewer.addCapsule(viewerPrefix+bodyName, radius,.8*length,color)
        gmodel.addGeometryObject(Capsule(jointName,jointId,pio.SE3.Identity(),1.5*radius,0.0))
        gmodel.addGeometryObject(Capsule(bodyName ,jointId,
                                         pio.SE3(eye(3),np.matrix([0.,0.,length/2]).T),
                                         radius,0.8*length))
        jointPlacement     = pio.SE3(eye(3),np.matrix([0.0,0.0,length]).T)

    rmodel.addFrame( pio.Frame('tip',jointId,0,jointPlacement,pio.FrameType.OP_FRAME) )

    rmodel.upperPositionLimit = np.zeros(nbJoint)+2*np.pi
    rmodel.lowerPositionLimit = np.zeros(nbJoint)-2*np.pi
    rmodel.velocityLimit      = np.zeros(nbJoint)+5.0
    
    return rmodel,gmodel

def createPendulumWrapper(nbJoint,initViewer=True):
    '''
    Returns a RobotWrapper with a N-pendulum inside.
    '''
    gv = GepettoViewerServer() if initViewer else None
    rmodel,gmodel = createPendulum(nbJoint,viewer=gv)
    rw = pio.RobotWrapper(rmodel,visual_model=gmodel,collision_model=gmodel)
    if initViewer: rw.initViewer(loadModel=True) 
    return rw

if __name__ == "__main__":
    rw = createPendulumWrapper(3,True)
    
