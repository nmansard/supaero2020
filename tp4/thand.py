import pinocchio as pio
pio.switchToNumpyMatrix()
from pinocchio.utils import rand,zero,rotate
from robot_hand import RobotHand
import time

robot = RobotHand()
robot.display(robot.q0)

q = robot.q0.copy()
robot.display(q)

rmodel = robot.model
rdata  = rmodel.createData()
gmodel = robot.gmodel
gdata  = gmodel.createData()
gdata.collisionRequest.enable_contact = True

gv = robot.viewer
#gv.setCameraTransform(0,[-0.14, 0.25, 0.22, -0.13, 0.43, 0.81, -0.38])

q[0]=-1#np.pi/2
q.flat[2:5] = [-.3,-.2,-.1]
vq = -rand(rmodel.nv)
vq[0]=0

for i in range(1000):
    q += vq*1e-2
    robot.display(q)
    time.sleep(1e-3)
    
    if pio.computeCollisions(rmodel,rdata,gmodel,gdata,q,False):
        break

    
for ir,r in enumerate(gdata.collisionResults):
    if r.isCollision():
        c=r.getContact(0)
        pair = gmodel.collisionPairs[ir]
        col1,col2 = pair.first,pair.second
        print(gmodel.geometryObjects[col1].name,gmodel.geometryObjects[col2].name) 
        robot.displayContact(r.getContact(0),refresh=True)

