import pinocchio as pio
import numpy as np
from pinocchio.utils import rand,zero,rotate

class CollisionWrapper:
    def __init__(self,robot):
        self.robot=robot
        self.viewer=robot.viewer

        self.rmodel = robot.model
        self.rdata  = self.rmodel.createData()
        self.gmodel = self.robot.gmodel
        self.gdata  = self.gmodel.createData()
        self.gdata.collisionRequest.enable_contact = True


    def computeCollisions(self,q,vq=None):
        res = pio.computeCollisions(self.rmodel,self.rdata,self.gmodel,self.gdata,q,False)
        pio.computeDistances(self.rmodel,self.rdata,self.gmodel,self.gdata,q)
        pio.computeJointJacobians(self.rmodel,self.rdata,q)
        if not(vq is None):
            pio.forwardKinematics(self.rmodel,self.rdata,q,vq,0*vq)
        return res

    def getCollisionList(self):
        '''Return a list of triplets [ index,collision,result ] where index is the
        index of the collision pair, colision is gmodel.collisionPairs[index]
        and result is gdata.collisionResults[index].
        '''
        return [ [ir,self.gmodel.collisionPairs[ir],r]
                 for ir,r in enumerate(self.gdata.collisionResults) if r.isCollision() ]

    def displayCollisions(self,collisions,refresh=False):
        '''Display in the viewer the collision list get from getCollisionList().'''
        if self.viewer is None: return

        for ic,[i,c,r] in enumerate(collisions):
            self.robot.displayContact(r.getContact(0),ic)
        self.robot.hideContact(len(collisions)) # Hide all other contacts from ic to robot.maxContact
        if refresh: self.viewer.refresh()

    def _getCollisionJacobian(self,col,res):
        '''Compute the jacobian for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pio.SE3(pio.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,contact.normal).matrix(),contact.pos)
    
        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse()*oMj1
        cMj2 = oMc.inverse()*oMj2
        
        J1=pio.getJointJacobian(self.rmodel,self.rdata,joint1,pio.ReferenceFrame.LOCAL)
        J2=pio.getJointJacobian(self.rmodel,self.rdata,joint2,pio.ReferenceFrame.LOCAL)
        Jc1=cMj1.action*J1
        Jc2=cMj2.action*J2
        J = (Jc1-Jc2)[2,:]
        return J

    def _getCollisionJdotQdot(self,col,res):
        '''Compute the Coriolis acceleration for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pio.SE3(pio.Quaternion.FromTwoVectors(np.matrix([0,0,1]).T,contact.normal).matrix(),contact.pos)
    
        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse()*oMj1
        cMj2 = oMc.inverse()*oMj2

        a1 = self.rdata.a[joint1]
        a2 = self.rdata.a[joint2]
        a = (cMj1*a1-cMj2*a2).linear[2]
        return a
        
    def getCollisionJacobian(self,collisions):
        '''From a collision list, return the Jacobian corresponding to the normal direction.  '''
        if len(collisions)==0: return np.ndarray([0,self.rmodel.nv])
        J = np.vstack([ self._getCollisionJacobian(c,r) for (i,c,r) in collisions ])
        return J

    def getCollisionJdotQdot(self,collisions):
        if len(collisions)==0: return np.matrix([]).T
        a0 = np.vstack([ self._getCollisionJdotQdot(c,r) for (i,c,r) in collisions ])
        return a0

    def getCollisionDistances(self,collisions):
        if len(collisions)==0: return np.matrix([]).T
        dist = np.matrix([ self.gdata.distanceResults[i].min_distance for (i,c,r) in collisions ]).T
        return dist
