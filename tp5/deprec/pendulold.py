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
from pinocchio.explog import exp,log
from numpy.linalg import pinv,norm
import pinocchio as pio
import gepetto.corbaserver
from numpy.linalg import pinv,norm,inv
import time

# --- WRAPPER AROUND DISPLAY ------------------------------------------------------------
# --- WRAPPER AROUND DISPLAY ------------------------------------------------------------
# --- WRAPPER AROUND DISPLAY ------------------------------------------------------------

# Example of a class Display that connect to Gepetto-viewer and implement a
# 'place' method to set the position/rotation of a 3D visual object in a scene.
class Display():
    '''
    Class Display: Example of a class implementing a client for the Gepetto-viewer server. The main
    method of the class is 'place', that sets the position/rotation of a 3D visual object in a scene.
    '''
    def __init__(self,windowName = "pinocchio" ):
        '''
        This function connect with the Gepetto-viewer server and open a window with the given name.
        If the window already exists, it is kept in the current state. Otherwise, the newly-created
        window is set up with a scene named 'world'.
        '''

        # Create the client and connect it with the display server.
        try:
            self.viewer=gepetto.corbaserver.Client()
        except:
            print("Error while starting the viewer client. ")
            print("Check whether Gepetto-viewer is properly started")

        # Open a window for displaying your model.
        try:
            # If the window already exists, do not do anything.
            windowID = self.viewer.gui.getWindowID (windowName)
            print("Warning: window '"+windowName+"' already created.")
            print("The previously created objects will not be destroyed and do not have to be created again.")
        except:
            # Otherwise, create the empty window.
            windowID = self.viewer.gui.createWindow (windowName)
            # Start a new "scene" in this window, named "world", with just a floor.
            self.viewer.gui.createScene("world")
            self.viewer.gui.addSceneToWindow("world",windowID)

        # Finally, refresh the layout to obtain your first rendering.
        self.viewer.gui.refresh()

    def nofloor(self):
        '''
        This function will hide the floor.
        '''
        self.viewer.gui.setVisibility('world/floor',"OFF")
        self.viewer.gui.refresh()

    def place(self,objName,M,refresh=True):
        '''
        This function places (ie changes both translation and rotation) of the object
        names "objName" in place given by the SE3 object "M". By default, immediately refresh
        the layout. If multiple objects have to be placed at the same time, do the refresh
        only at the end of the list.
        '''
        self.viewer.gui.applyConfiguration(objName,
                                           se3ToXYZQUAT(M))
        if refresh: self.viewer.gui.refresh()


class Visual:
    '''
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
    * the name of the 3D objects inside Gepetto viewer.
    * the ID of the joint in the kinematic tree to which the body is attached.
    * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    '''
    def __init__(self,name,jointParent,placement):
        self.name = name                  # Name in gepetto viewer
        self.jointParent = jointParent    # ID (int) of the joint 
        self.placement = placement        # placement of the body wrt joint, i.e. bodyMjoint
    def place(self,display,oMjoint):
        oMbody = oMjoint*self.placement
        display.place(self.name,oMbody,False)


# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
        
class Pendulum:
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

    def __init__(self,nbJoint=1):
        '''Create a Pinocchio model of a N-pendulum, with N the argument <nbJoint>.'''
        self.viewer     = Display()
        self.visuals    = []
        self.model      = pio.Model()
        self.createPendulum(nbJoint)
        self.data       = self.model.createData()

        self.q0         = zero(self.model.nq)

        self.DT         = 5e-2   # Step length
        self.NDT        = 2      # Number of Euler steps per integration (internal)
        self.Kf         = .10    # Friction coefficient
        self.vmax       = 8.0    # Max velocity (clipped if larger)
        self.umax       = 2.5    # Max torque   (clipped if larger)
        self.withSinCos = False   # If true, state is [cos(q),sin(q),qdot], else [q,qdot]

    def createPendulum(self,nbJoint,rootId=0,prefix='',jointPlacement=None):
        color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred = [1.0,0.0,0.0,1.0]

        jointId = rootId
        jointPlacement     = jointPlacement if jointPlacement!=None else pio.SE3.Identity()
        length = 1.0
        mass = length
        inertia = pio.Inertia(mass,
                              np.matrix([0.0,0.0,length/2]).T,
                              mass/5*np.diagflat([ 1e-2,length**2,  1e-2 ]) )

        for i in range(nbJoint):
            istr = str(i)
            name               = prefix+"joint"+istr
            jointName,bodyName = [name+"_joint",name+"_body"]
            jointId = self.model.addJoint(jointId,pio.JointModelRY(),jointPlacement,jointName)
            self.model.appendBodyToJoint(jointId,inertia,pio.SE3.Identity())
            try:self.viewer.viewer.gui.addSphere('world/'+prefix+'sphere'+istr, 0.15,colorred)
            except: pass
            self.visuals.append( Visual('world/'+prefix+'sphere'+istr,jointId,pio.SE3.Identity()) )
            try:self.viewer.viewer.gui.addCapsule('world/'+prefix+'arm'+istr, .1,.8*length,color)
            except:pass
            self.visuals.append( Visual('world/'+prefix+'arm'+istr,jointId,
                                        pio.SE3(eye(3),np.matrix([0.,0.,length/2]).T)))
            jointPlacement     = pio.SE3(eye(3),np.matrix([0.0,0.0,length]).T)

        self.model.addFrame( pio.Frame('tip',jointId,0,jointPlacement,pio.FrameType.OP_FRAME) )

    def display(self,q):
        pio.forwardKinematics(self.model,self.data,q)
        for visual in self.visuals:
            visual.place( self.viewer,self.data.oMi[visual.jointParent] )
        self.viewer.viewer.gui.refresh()


    @property
    def nq(self): return self.model.nq
    @property
    def nv(self): return self.model.nv
    @property
    def nx(self): return self.nq+self.nv
    @property
    def nobs(self): return self.nx+self.withSinCos
    @property
    def nu(self): return self.nv

    def reset(self,x0=None):
        if x0 is None: 
            q0 = np.pi*(rand(self.nq)*2-1)
            v0 = rand(self.nv)*2-1
            x0 = np.vstack([q0,v0])
        assert len(x0)==self.nx
        self.x = x0.copy()
        self.r = 0.0
        return self.obs(self.x)

    def step(self,u):
        assert(len(u)==self.nu)
        _,self.r = self.dynamics(self.x,u)
        return self.obs(self.x),self.r

    def obs(self,x):
        if self.withSinCos:
            return np.vstack([ np.vstack([np.cos(qi),np.sin(qi)]) for qi in x[:self.nq] ] 
                             + [x[self.nq:]],)
        else: return x.copy()

    def tip(self,q):
        '''Return the altitude of pendulum tip'''
        pio.framesKinematics(self.model,self.data,q)
        return self.data.oMf[1].translation[2,0]

    def dynamics(self,x,u,display=False,verbose=False):
        '''
        Dynamic function: x,u -> xnext=f(x,y).
        Put the result in x (the initial value is destroyed). 
        Also compute the cost of making this step.
        Return x for convenience along with the cost.
        '''

        modulePi = lambda th: (th+np.pi)%(2*np.pi)-np.pi
        sumsq    = lambda x : np.sum(np.square(x))

        cost = 0.0
        q = modulePi(x[:self.nq])
        v = x[self.nq:]
        u = np.clip(np.reshape(np.matrix(u),[self.nu,1]),-self.umax,self.umax)


        DT = self.DT/self.NDT
        for i in range(self.NDT):
            #pio.computeAllTerms(self.model,self.data,q,v)
            #M   = np.matrix(self.data.M)
            #b   = np.matrix(self.data.nle).T
            tau = u-self.Kf*v
            #print(M,u,self.Kf,v,b)
            #a   = inv(M)*(u-self.Kf*v-b)
            a = pio.aba(self.model,self.data,q,v,tau)
            if verbose: print(q,v,tau,a)
            
            v    += a*DT
            q    += v*DT
            cost += (sumsq(q) + 1e-1*sumsq(v) + 1e-3*sumsq(u))*DT

            if display:
                self.display(q)
                time.sleep(1e-4)

        x[:self.nq] = modulePi(q)
        x[self.nq:] = np.clip(v,-self.vmax,self.vmax)
        
        return x,-cost
     
    def render(self):
        q = self.x[:self.nq]
        self.display(q)
        time.sleep(self.DT/10)

        
# -----------------------------------------------------------------------
# --- DISCRETE PENDULUM -------------------------------------------------
NQ   = 21  # Discretization steps for position
NV   = 19  # Discretization steps for velocity
VMAX = 5   # Max velocity (v in [-vmax,vmax])
NU   = 9   # Discretization steps for torque
UMAX = 2.5  # Max torque (u in [-umax,umax])
DT   = 3e-1

DQ = 2*np.pi/NQ
DV = 2.0*(VMAX)/NV
DU = 2.0*(UMAX)/NU

# Continuous to discrete
def c2dq(q):
    q = (q+np.pi)%(2*np.pi)
    return int(round(q/DQ-.5))  % NQ

def c2dv(v):
    v = np.clip(v,-VMAX+1e-3,VMAX-1e-3)
    return int(np.floor((v+VMAX)/DV))

def c2du(u):
    u = np.clip(u,-UMAX+1e-3,UMAX-1e-3)
    return int(np.floor((u+UMAX)/DU))

def c2d(qv):
    '''From continuous to discrete.'''
    return c2dq(qv[0]),c2dv(qv[1])

# Discrete to continuous
def d2cq(iq):  ### q = iq*DQ+DQ/2     ...   (q-DQ/2)/DQ = q/DQ-1/2
    iq = np.clip(iq,0,NQ-1)
    return iq*DQ - np.pi +DQ/2

def d2cv(iv):
    iv = np.clip(iv,0,NV-1) - (NV-1)/2
    return iv*DV

def d2cu(iu):
    iu = np.clip(iu,0,NU-1) - (NU-1)/2
    return iu*DU
    

    
def d2c(iqv):
    '''From discrete to continuous'''
    return d2cq(iqv[0]),d2cv(iqv[1])

def x2i(x): return x[0]*NV+x[1]
def i2x(i): return [ i//NV, i%NV ]

# --- PENDULUM

class DiscretePendulum:
    '''
    Fully discrete pendulum with X \in [0..NX-1] and U \in [0..NU-1]. 
    '''
    def __init__(self):
        self.pendulum = Pendulum(1)
        self.pendulum.DT  = DT
        self.pendulum.NDT = 5

    @property
    def nqv(self): return [NQ,NV]
    @property
    def nx(self): return NQ*NV
    @property
    def nu(self): return NU
    @property
    def goal(self): return x2i(c2d([0.,0.]))

    def reset(self,x=None):
        if x is None:
            x = [ np.random.randint(0,NQ), np.random.randint(0,NV) ]
        else: x = i2x(x)
        assert(len(x)==2)
        self.x = x
        return x2i(self.x)

    def step(self,iu):
        self.x     = self.dynamics(self.x,iu)
        reward     = 1 if x2i(self.x)==self.goal else 0
        return x2i(self.x),reward

    def render(self):
        q = d2cq(self.x[0])
        self.pendulum.display(np.matrix([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self,ix,iu,verbose=False):
        x   = np.matrix(d2c (ix)).T
        u   = d2cu(iu)
        
        self.xc,_ = self.pendulum.dynamics(x,u,verbose=verbose)
        return c2d(x.T.tolist()[0])
        
class HybridPendulum(object):
    '''
    Semi-discrete pendulum, with digitalized control U \in [0..NU-1] and
    continuous state X \in R^2.
    '''
    def __init__(self):
        self.pendulum = Pendulum(1)
        self.pendulum.DT  = DT
        self.pendulum.NDT = 5

    @property
    def nx(self): return self.pendulum.nx
    @property
    def nu(self): return NU
    @property
    def nobs(self): return self.pendulum.nobs
    @property
    def x(self): return self.pendulum.x
    
    @property
    def withSinCos(self):     return self.pendulum.withSinCos
    @withSinCos.setter
    def withSinCos(self,b):   self.pendulum.withSinCos=b

    def reset(self,x=None):
        return self.pendulum.reset()

    def step(self,iu):
        u   = np.matrix([d2cu(iu)])
        return self.pendulum.step(u)
        
    def render(self):
        self.pendulum.render()

    def dynamics(self,x,iu):
        u   = d2cu(iu)
        x,_ = self.pendulum.dynamics(x,u)
        return x

    @property
    def obs(self,x=None):
        x = self.pendulum.x if x is None else x
        return self.pendulum.obs(x)
