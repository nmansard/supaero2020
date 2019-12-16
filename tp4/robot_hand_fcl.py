from pinocchio.utils import *
from pinocchio.explog import exp,log
from numpy.linalg import pinv,norm
import pinocchio as pio
import gepetto.corbaserver
from numpy import pi
from numpy import cos,sin,pi,hstack,vstack,argmin
from numpy.linalg import norm,pinv

from gviewserver import GepettoViewerServer

class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).

    The visual are supposed mostly to be capsules. In that case, the object also contains
    radius and length of the capsule.
    The collision checking computes collision test, distance, and witness points.
    Using the supporting robot, the collision Jacobian returns a 1xN matrix corresponding
    to the normal direction.
    '''
    def __init__(self,name,jointParent,placement,radius=.1,length=None):
        '''Length and radius are used in case of capsule objects'''
        self.name = name                  # Name in gepetto viewer
        self.jointParent = jointParent    # ID (int) of the joint 
        self.placement = placement        # placement of the body wrt joint, i.e. bodyMjoint
        if length is not None:
            self.length = length
            self.radius = radius

    def place(self,gview,oMjoint,refresh=True):
        oMbody = oMjoint*self.placement
        #gview.place(self.name,oMbody,False)
        gview.applyConfiguration(self.name,
                                   se3ToXYZQUAT(oMbody))
        if refresh: gview.refresh()

    def isCapsule(self):
        return 'length' in self.__dict__ and 'radius' in self.__dict__


def Capsule(name,joint,placement,radius,length):
    caps = pio.GeometryObject.CreateCapsule(radius,length)
    caps.name = name
    caps.placement = placement
    caps.parentJoint = joint
    return caps


class FakeViewer:
    def addCapsule(self,*args):        pass
    def addSphere(self,*args):        pass
    def addCylinder(self,*args):        pass
    def setVisibility(self,*args):        pass
    def applyConfiguration(self,*args):        pass
    def refresh(self,*args):        pass
        
class Robot:
    '''
    Define a class Robot with 7DOF (shoulder=3 + elbow=1 + wrist=3). 
    The configuration is nq=7. The velocity is the same. 
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).
    
    CollisionPairs is a list of visual indexes. 
    Reference to the collision pair is used in the collision test and jacobian of the collision
    (which are simply proxy method to methods of the visual class).
    '''

    def __init__(self):
        self.viewer = GepettoViewerServer()
        if self.viewer is None: self.viewer = FakeViewer()
        #self.visuals = []
        self.model = pio.Model()
        self.gmodel = pio.GeometryModel()

        self.createHand()
        self.addCollisionPairs()

        self.data = self.model.createData()
        self.gdata = pio.GeometryData(self.gmodel)
        
        self.q0 = zero(self.model.nq)
        #self.q0[3] = 1.0
        self.v0 = zero(self.model.nv)
        self.collisionPairs = []

    def addCollisionPairs(self):
        # self.gmodel.addAllCollisionPairs()
        # pairs = self.gmodel.collisionPairs
        # parents = self.model.parents
        # gobjs = self.gmodel.geometryObjects
        # wp = [ p for p in pairs \
        #        if parents[gobjs[p.first ].parentJoint] == gobjs[p.second].parentJoint \
        #        or parents[gobjs[p.second].parentJoint] == gobjs[p.first ].parentJoint ]
        # for p in reversed(wp): self.gmodel.removeCollisionPair(p)     
        for n in [ 'world/finger11', 'world/finger12', 'world/finger13' ]:
            self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
                                                          self.gmodel.getGeometryId('world/wpalmr')))
            # self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
            #                                                self.gmodel.getGeometryId('world/palm2')))

        self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/finger12'),
                                                       self.gmodel.getGeometryId('world/palm2')))
        self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/finger13'),
                                                       self.gmodel.getGeometryId('world/palm2')))

        # self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/finger11'),
        #                                                self.gmodel.getGeometryId('world/finger12')))
        # self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/finger12'),
        #                                                self.gmodel.getGeometryId('world/finger13')))
        
        
    def addCapsule(self,name,joint,placement,r,l):
        self.gmodel.addGeometryObject(Capsule(name,joint,placement,r*0.99,l))

    def createHand(self,rootId=0,prefix='',jointPlacement=None):
        color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred = [1.0,0.0,0.0,1.0]

        jointId = rootId

        cm = 1e-2
        trans = lambda x,y,z: pio.SE3(eye(3),np.matrix([x,y,z]).T)
        inertia = lambda m,c: pio.Inertia(m,np.matrix(c,np.double).T,eye(3)*m**2)

        name               = prefix+"wrist"
        jointName,bodyName = [name+"_joint",name+"_body"]
        #jointPlacement     = jointPlacement if jointPlacement!=None else pio.SE3.Identity()
        jointPlacement     = jointPlacement if jointPlacement!=None else pio.SE3(pio.utils.rotate('y',np.pi),zero(3))
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(3,[0,0,0]),pio.SE3.Identity())
        
        L=3*cm;W=5*cm;H=1*cm
        #self.viewer.addSphere('world/'+prefix+'wrist', .02, color)
        self.viewer.addCapsule('world/'+prefix+'wrist', .02, 0, color)
        #self.visuals.append( Visual('world/'+prefix+'wpalmt',jointId,
        #                            pio.SE3(rotate('x',pi/2),np.matrix([0,0,0]).T),.02,0 ))
        self.addCapsule('world/'+prefix+'wrist',jointId,
                        pio.SE3(rotate('x',pi/2),np.matrix([0,0,0]).T),.02,0 )
  
        # self.viewer.addBox('world/'+prefix+'wpalm', L/2, W/2, H,color)
        # self.visuals.append( Visual('world/'+prefix+'wpalm',jointId,trans(L/2,0,0) ))
        capsr = H; capsl = W        
        # for IPOS,POS in enumerate(np.arange(0,L+1e-3,L/10)):
        #     name = 'world/'+prefix+'wpalmb%02d'%IPOS
        #     self.viewer.addCapsule(name, capsr, capsl, color)
        #     #self.visuals.append( Visual(name,jointId,
        #     #pio.SE3(rotate('x',pi/2),np.matrix([POS,0,0]).T),
        #     #                            capsr,capsl ))
        #     self.addCapsule(name,jointId,
        #                     pio.SE3(rotate('x',pi/2),np.matrix([POS,0,0]).T),capsr,capsl )
 
        capsr = H; capsl = L
        self.viewer.addCapsule('world/'+prefix+'wpalml', H, L, color)
        #self.visuals.append( Visual('world/'+prefix+'wpalml',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([L/2,-W/2,0]).T),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'wpalml',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([L/2,-W/2,0]).T),capsr,capsl )
 
        capsr = H; capsl = L
        self.viewer.addCapsule('world/'+prefix+'wpalmr', H, L, color)
        #self.visuals.append( Visual('world/'+prefix+'wpalmr',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([L/2,W/2,0]).T),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'wpalmr',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([L/2,W/2,0]).T),capsr,capsl )
        print('Dimension capsule wpalmr %.3f, %.3f',(capsr,capsl))
        

        name               = prefix+"palm"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([5*cm,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(2,[0,0,0]),pio.SE3.Identity())
        capsr = 1*cm; capsl = W
        self.viewer.addCapsule('world/'+prefix+'palm2', 1*cm, W, color)
        #self.visuals.append( Visual('world/'+prefix+'palm2',jointId,
        #                            pio.SE3(rotate('x',pi/2),zero(3)),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'palm2',jointId,
                                    pio.SE3(rotate('x',pi/2),zero(3)),capsr,capsl )
 

        FL = 4*cm
        palmIdx = jointId

        name               = prefix+"finger11"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,W/2,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger11', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger11',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'finger11',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )
        print('Dimension capsule finger11 %.3f, %.3f',(capsr,capsl))


        name               = prefix+"finger12"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger12', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger12',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'finger12',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )


        name               = prefix+"finger13"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-2*H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = 0.;
        self.viewer.addSphere('world/'+prefix+'finger13', capsr, color)
        ##self.visuals.append( Visual('world/'+prefix+'finger13',jointId,
        #                           trans(2*H,0,0),capsr,capsl )
#)
        self.addCapsule('world/'+prefix+'finger13',jointId,
                                    trans(2*H,0,0),capsr,capsl )


        name               = prefix+"finger21"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,0,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger21', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger21',jointId,
        #pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )
        #)
        self.addCapsule('world/'+prefix+'finger21',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )


        name               = prefix+"finger22"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger22', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger22',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )
        #)
        self.addCapsule('world/'+prefix+'finger22',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )


        name               = prefix+"finger23"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = 0.;
        self.viewer.addSphere('world/'+prefix+'finger23', H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger23',jointId,
        #                            trans(H,0,0),capsr,capsl )
        #)
        self.addCapsule('world/'+prefix+'finger23',jointId,
                                    trans(H,0,0),capsr,capsl )



        name               = prefix+"finger31"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,-W/2,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger31', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger31',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl ))
        self.addCapsule('world/'+prefix+'finger31',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )


        name               = prefix+"finger32"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'finger32', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger32',jointId,
        #                            pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl ))
        self.addCapsule('world/'+prefix+'finger32',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),capsr,capsl )


        name               = prefix+"finger33"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-2*H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = 0.
        self.viewer.addSphere('world/'+prefix+'finger33', H, color)
        #self.visuals.append( Visual('world/'+prefix+'finger33',jointId,
        #                            trans(2*H,0,0),capsr,capsl ))
        self.addCapsule('world/'+prefix+'finger33',jointId,
                                    trans(2*H,0,0),capsr,capsl )


        name               = prefix+"thumb1"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([1*cm,-W/2-H*1.5,0]).T)
        jointId = self.model.addJoint(1,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = 2*cm
        self.viewer.addCapsule('world/'+prefix+'thumb1', H, 2*cm, color)
        #self.visuals.append( Visual('world/'+prefix+'thumb1',jointId,
        #                            pio.SE3(rotate('z',pi/3)*rotate('x',pi/2),np.matrix([1*cm,-1*cm,0]).T),
        #capsr,capsl ))
        self.addCapsule('world/'+prefix+'thumb1',jointId,
                        pio.SE3(rotate('z',pi/3)*rotate('x',pi/2),np.matrix([1*cm,-1*cm,0]).T),
                        capsr,capsl )
        
        name               = prefix+"thumb2"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(rotate('z',pi/3)*rotate('x',pi), np.matrix([3*cm,-1.8*cm,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRZ(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.4,[0,0,0]),pio.SE3.Identity())
        capsr = H; capsl = FL-2*H
        self.viewer.addCapsule('world/'+prefix+'thumb2', H, FL-2*H, color)
        #self.visuals.append( Visual('world/'+prefix+'thumb2',jointId,
        #                            pio.SE3(rotate('x',pi/3),np.matrix([-0.7*cm,.8*cm,-0.5*cm]).T),
        #capsr,capsl ))
        self.addCapsule('world/'+prefix+'thumb2',jointId,
                        pio.SE3(rotate('x',pi/3),np.matrix([-0.7*cm,.8*cm,-0.5*cm]).T),
                        capsr,capsl )

        # Prepare some patches to represent collision points. Yet unvisible.
        for i in range(10):
            self.viewer.addCylinder('world/wa'+str(i), .01, .003, [ 1.0,0,0,1])
            self.viewer.addCylinder('world/wb'+str(i), .01, .003, [ 1.0,0,0,1])
            self.viewer.setVisibility('world/wa'+str(i),'OFF')
            self.viewer.setVisibility('world/wb'+str(i),'OFF')

  
        
    def display(self,q):
        pio.forwardKinematics(self.model,self.data,q)
        pio.updateGeometryPlacements(self.model,self.data,self.gmodel,self.gdata)
        for i,g in enumerate(self.gmodel.geometryObjects):
            self.viewer.applyConfiguration(g.name, pio.se3ToXYZQUATtuple(self.gdata.oMg[i]))
        self.viewer.refresh()


