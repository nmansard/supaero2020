from pinocchio.utils import *
from pinocchio.explog import exp,log
from numpy.linalg import pinv,norm
import pinocchio as pio
import gepetto.corbaserver
from numpy import pi
from numpy import cos,sin,pi,hstack,vstack,argmin
from numpy.linalg import norm,pinv
import hppfcl
from gviewserver import GepettoViewerServer

def Capsule(name,joint,placement,radius,length):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    caps = pio.GeometryObject.CreateCapsule(radius,length)
    caps.name = name
    caps.placement = placement
    caps.parentJoint = joint
    return caps

class RobotHand:
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
        #self.visuals = []
        self.model = pio.Model()
        self.gmodel = pio.GeometryModel()

        self.createHand()
        self.addCollisionPairs()

        self.data = self.model.createData()
        self.gdata = pio.GeometryData(self.gmodel)
        self.gdata.collisionRequest.enable_contact=True
        
        self.q0 = zero(self.model.nq)
        self.q0[0] = np.pi
        self.q0[-2] = -np.pi/3
        self.q0[2:-4] = -np.pi/6
        self.v0 = zero(self.model.nv)
        self.collisionPairs = []

    def addCollisionPairs(self):
        # # Collision between finger 1 and right palm
        # for nph in range(1,4):
        #     n= 'world/finger%d%d' % (1,nph)
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
        #                                                    self.gmodel.getGeometryId('world/wpalmr')))
        # # Collision between finger 2 and wirst
        # for nph in range(1,4):
        #     n= 'world/finger%d%d' % (2,nph)
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
        #                                                    self.gmodel.getGeometryId('world/wrist')))
        # # Collision between finger 3 and left palm
        # for nph in range(1,4):
        #     n= 'world/finger%d%d' % (3,nph)
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
        #                                                    self.gmodel.getGeometryId('world/wpalml')))
            
        # # Collision between phallange 2 and 3 and second palm
        # for nf in range(1,4):
        #     for nph in range(2,4):
        #         n= 'world/finger%d%d' % (nf,nph)
        #         self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
        #                                                        self.gmodel.getGeometryId('world/palm2')))
        #         self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n),
        #                                                        self.gmodel.getGeometryId('world/wpalmfr')))

        #  # Collision between phallange 1 and phallange 3
        # for nf in range(1,4):
        #     n1= 'world/finger%d%d' % (nf,1)
        #     n2= 'world/finger%d%d' % (nf,3)
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(n1),
        #                                                    self.gmodel.getGeometryId(n2)))

        # self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/wpalml'),
        #                                                self.gmodel.getGeometryId('world/thumb2')))
        # self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/wpalml'),
        #                                                self.gmodel.getGeometryId('world/thumb2')))
        # for it in range(1,3):
        #     nt= 'world/thumb%d' % it
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/wpalmfr'),
        #                                                    self.gmodel.getGeometryId(nt)))
        #     self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/palm2'),
        #                                                    self.gmodel.getGeometryId(nt)))
        #     for iph in range(1,4):
        #         nph= 'world/finger%d%d' % (3,iph)
        #         self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId(nt),
        #                                                        self.gmodel.getGeometryId(nph)))
        
        pairs = [
            ['finger11','wpalmr'],
            ['finger12','wpalmr'],
            ['finger13','wpalmr'],
            ['finger21','wrist'],
            ['finger22','wrist'],
            ['finger23','wrist'],
            ['finger31','wpalml'],
            ['finger32','wpalml'],
            ['finger33','wpalml'],
            ['finger12','palm2'],
            ['finger12','wpalmfr'],
            ['finger13','palm2'],
            ['finger13','wpalmfr'],
            ['finger22','palm2'],
            ['finger22','wpalmfr'],
            ['finger23','palm2'],
            ['finger23','wpalmfr'],
            ['finger32','palm2'],
            ['finger32','wpalmfr'],
            ['finger33','palm2'],
            ['finger33','wpalmfr'],
            ['finger11','finger13'],
            ['finger21','finger23'],
            ['finger31','finger33'],
            ['wpalml','thumb2'],
            ['wpalmfr','thumb1'],
            ['palm2','thumb1'],
            ['thumb1','finger31'],
            ['thumb1','finger31'],
            ['thumb1','finger33'],
            ['wpalmfr','thumb2'],
            ['palm2','thumb2'],
            ['thumb2','finger31'],
            ['thumb2','finger32'],
            ['thumb2','finger33'],
        ]
        for (n1,n2) in pairs:
            self.gmodel.addCollisionPair(pio.CollisionPair(self.gmodel.getGeometryId('world/'+n1),
                                                           self.gmodel.getGeometryId('world/'+n2)))

        
    def addCapsule(self,name,joint,placement,radius,length,color=[1,1,0.78,1]):
        self.gmodel.addGeometryObject(Capsule(name,joint,placement,radius*0.99,length))
        if self.viewer is not None: self.viewer.addCapsule(name, radius, length, color)

        
    def createHand(self,rootId=0,jointPlacement=None):
        color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred = [1.0,0.0,0.0,1.0]

        jointId = rootId

        cm = 1e-2
        trans = lambda x,y,z: pio.SE3(eye(3),np.matrix([x,y,z]).T)
        inertia = lambda m,c: pio.Inertia(m,np.matrix(c,np.double).T,eye(3)*m**2)

        name               = "wrist"
        jointName,bodyName = [name+"_joint",name+"_body"]
        #jointPlacement     = jointPlacement if jointPlacement!=None else pio.SE3.Identity()
        jointPlacement     = jointPlacement if jointPlacement!=None else pio.SE3(pio.utils.rotate('y',np.pi),zero(3))
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(3,[0,0,0]),pio.SE3.Identity())

        ## Hand dimsensions: length, width, height(=depth), finger-length
        L=3*cm;W=5*cm;H=1*cm; FL = 4*cm
        self.addCapsule('world/wrist',jointId,
                        pio.SE3(rotate('x',pi/2),np.matrix([0,0,0]).T),.02,0 )
  
        self.addCapsule('world/wpalml',jointId,
                                    pio.SE3(rotate('z',-.3)*rotate('y',pi/2),np.matrix([L/2,-W/2.6,0]).T),H,L )
        #pio.SE3(rotate('y',pi/2),np.matrix([L/2,-W/2,0]).T),H,L )
        self.addCapsule('world/wpalmr',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([L/2,W/2,0]).T),H,L)
        self.addCapsule('world/wpalmfr',jointId,
                                    pio.SE3(rotate('x',pi/2),np.matrix([L,0,0]).T),H,W)
        
        name               = "palm"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([5*cm,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(2,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/palm2',jointId,
        #                pio.SE3(rotate('y',pi/2),zero(3)),1*cm,W )
                        pio.SE3(rotate('x',pi/2),zero(3)),1*cm,W )
        palmIdx = jointId

        name               = "finger11"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,W/2,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger11',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H )

        name               = "finger12"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger12',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H )


        name               = "finger13"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-2*H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger13',jointId,
                                    trans(2*H,0,0),H,0 )

        name               = "finger21"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,0,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger21',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H )

        name               = "finger22"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger22',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H )

        name               = "finger23"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger23',jointId,
                                    trans(H,0,0),H,0 )

        name               = "finger31"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([2*cm,-W/2,0]).T)
        jointId = self.model.addJoint(palmIdx,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger31',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H)

        name               = "finger32"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger32',jointId,
                                    pio.SE3(rotate('y',pi/2),np.matrix([FL/2-H,0,0]).T),H,FL-2*H)

        name               = "finger33"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), np.matrix([FL-2*H,0,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/finger33',jointId,
                                    trans(2*H,0,0),H,0)

        name               = "thumb1"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(rotate('z',-1), np.matrix([1*cm,-W/2-H*1.3,0]).T)
        jointId = self.model.addJoint(1,pio.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        # self.addCapsule('world/thumb1',jointId,
        #                 pio.SE3(rotate('z',pi/3)*rotate('x',pi/2),np.matrix([1*cm,-1*cm,0]).T),
        #                 H,2*cm)
        
        name               = "thumb1a"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(eye(3), zero(3))
        jointId = self.model.addJoint(jointId,pio.JointModelRX(),jointPlacement,jointName)
        # self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/thumb1',jointId,
                        pio.SE3(rotate('z',pi/3)*rotate('x',pi/2),np.matrix([0.3*cm,-1.0*cm,0]).T),
                        H,2*cm)
        
        name               = "thumb2"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pio.SE3(rotate('z',pi/3)*rotate('x',pi), np.matrix([3*cm,-1.8*cm,0]).T)
        jointId = self.model.addJoint(jointId,pio.JointModelRZ(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.4,[0,0,0]),pio.SE3.Identity())
        self.addCapsule('world/thumb2',jointId,
                        pio.SE3(rotate('x',pi/3),np.matrix([-0.7*cm,.8*cm,-0.5*cm]).T),
                        H,FL-2*H)

        # Prepare some patches to represent collision points. Yet unvisible.
        if self.viewer is not None:
            self.maxContact = 10
            for i in range(self.maxContact):
                self.viewer.addCylinder('world/cpatch%d'%i, .01, .003, [ 1.0,0,0,1])
                self.viewer.setVisibility('world/cpatch%d'%i,'OFF')
    
    def hideContact(self,fromContactRef=0):
        if fromContactRef<0: fromContactRef=self.maxContact+fromContactRef
        for i in range(fromContactRef,self.maxContact):
            name='world/cpatch%d'%i
            self.viewer.setVisibility(name,'OFF')
    def displayContact(self,contact,contactRef=0,refresh=False):
        '''
        Display a small red disk at the position of the contact, perpendicular to the
        contact normal. 
        
        @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
        geomModel.collisionResults[0].geContact(0).
        @param contactRef: use patch named "world/cparch%d" % contactRef, 0 by default.
        @param refresh: option to refresh the viewer before returning, False by default.
        '''
        name='world/cpatch%d'%contactRef
        if self.viewer is None: return
        self.viewer.setVisibility(name,'ON')
        M = pio.SE3(pio.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,contact.normal).matrix(),contact.pos)
        self.viewer.applyConfiguration(name,pio.se3ToXYZQUATtuple(M))
        if refresh: self.viewer.refresh()   
        
    def display(self,q):
        pio.forwardKinematics(self.model,self.data,q)
        pio.updateGeometryPlacements(self.model,self.data,self.gmodel,self.gdata)
        if self.viewer is None: return
        for i,g in enumerate(self.gmodel.geometryObjects):
            self.viewer.applyConfiguration(g.name, pio.se3ToXYZQUATtuple(self.gdata.oMg[i]))
        self.viewer.refresh()
