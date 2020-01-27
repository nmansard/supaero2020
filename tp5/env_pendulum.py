'''
Create a simulation environment for a N-pendulum.
See the main at the end of the file for an example.
'''

import pinocchio as pio
import numpy as np
from models.pendulum import createPendulumWrapper
from env_abstract import EnvPinocchio
import env_abstract

# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
        
class EnvPendulum(EnvPinocchio):
    '''
    Define a class Robot with 7DOF (shoulder=3 + elbow=1 + wrist=3). 
    The configuration is nq=7. The velocity is the same. 
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).
    
    See tp1.py for an example of use.
    '''

    def __init__(self,nbJoint=1,withGepettoViewer=True):
        '''Create a Pinocchio model of a N-pendulum, with N the argument <nbJoint>.'''
        self.robot_wrapper = createPendulumWrapper(nbJoint,initViewer=withGepettoViewer)

        self.q0         = np.zeros(self.robot_wrapper.nq)
        self.v0         = np.zeros(self.robot_wrapper.nv)
        self.x0         = np.concatenate([self.q0,self.v0])

        EnvPinocchio.__init__(self,self.robot_wrapper.model,self.robot_wrapper,taumax=2.5)
        self.DT = 1e-2
        self.NDT = 5
        self.Kf = 1.0

        self.costWeights = { 'q': 1, 'v' : 1e-1, 'u' : 1e-3, 'tip' : 0. }
        self.tipDes = float(nbJoint)
        
    def cost(self,x=None,u=None):
        if x is None: x = self.x
        cost  = 0.
        q,v = x[:self.nq],x[-self.nv:]
        qdes = self.xdes[:self.nq]
        cost += self.costWeights['q']*np.sum((q-qdes)**2)
        cost += self.costWeights['v']*np.sum(v**2)
        cost += 0 if u is None else self.costWeights['u']*np.sum(u**2)
        cost += self.costWeights['tip']*(self.tip(q)-self.tipDes)**2
        return cost

    def tip(self,q=None):
        '''Return the altitude of pendulum tip'''
        if q is None: q = self.x[:self.nq]
        pio.framesForwardKinematics(self.rmodel,self.rdata,q)
        return self.rdata.oMf[1].translation[2]

# --- SPIN-OFF ----------------------------------------------------------------------------------
# --- SPIN-OFF ----------------------------------------------------------------------------------
# --- SPIN-OFF ----------------------------------------------------------------------------------

class EnvPendulumDiscrete(env_abstract.EnvDiscretized):
    def __init__(self,nbJoint=1,withGepettoViewer=True):
        env = EnvPendulum(nbJoint,withGepettoViewer)
        env.DT=5e-1
        env.NDT=5
        env.Kf=0.1
        env_abstract.EnvDiscretized.__init__(self,env,21,11)
        self.discretize_x.modulo = np.pi*2
        self.discretize_x.moduloIdx = range(env.nq)
        self.discretize_x.vmax[:env.nq] = np.pi
        self.discretize_x.vmin[:env.nq] = -np.pi
        self.reset()
        self.conti.costWeights = { 'q': 0, 'v' : 0, 'u' : 0, 'tip' : 1 }
        self.withSimpleCost = True
    def step(self,u):
        x,c=env_abstract.EnvDiscretized.step(self,u)
        if self.withSimpleCost:
            c = int(np.all(np.abs(self.conti.x)<1e-3))
        return x,c
        
class EnvPendulumSinCos(env_abstract.EnvPartiallyObservable):
    def __init__(self,nbJoint=1,withGepettoViewer=True):
        env = EnvPendulum(nbJoint,withGepettoViewer)
        def sincos(x,nq):
            q,v = x[:nq],x[nq:]
            return np.concatenate([np.concatenate([(np.cos(qi),np.sin(qi)) for qi in q]),v])
        def atan(x,nq):
            cq,sq,v = x[:2*nq:2],x[1:2*nq,2],x[2*nq:]
            return np.concatenate([np.arctan2(sq,cq),v])
        env_abstract.EnvPartiallyObservable.__init__(self,env,
                                                     lambda x:sincos(x,env.nq),
                                                     lambda csv:atan(csv,nq))
        self.reset()
        
class EnvPendulumHybrid(env_abstract.EnvDiscretized):
    def __init__(self,nbJoint=1,withGepettoViewer=True):
        env = EnvPendulumSinCos(nbJoint,withGepettoViewer)
        env_abstract.EnvDiscretized.__init__(self,env,discretize_x=0,discretize_u=11)
        self.reset()
        
        
# --- MAIN -------------------------------------------------------------------------------
# --- MAIN -------------------------------------------------------------------------------
# --- MAIN -------------------------------------------------------------------------------

if __name__ == "__main__":
    env = EnvPendulum(1)
    env.reset()
    for i in range(10):
        env.step(np.zeros(env.nu))
        env.render()      

    env = EnvPendulumDiscrete(1)
    u0 = env.encode_u(np.zeros(1)-1.9)
    env.reset(10)
    for i in range(10):
        env.step(u0)
        env.render()

    env = EnvPendulumSinCos(1)
    env.reset()

    env = EnvPendulumHybrid(1)
    env.reset()
    env.step(0)
