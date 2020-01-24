import pinocchio as pio
pio.switchToNumpyMatrix()
from example_robot_data import loadSolo,getModelPath,readParamsFromSrdf

def reduceGeomModel(model,data,modelReduced,dataReduced,geomModel,geomData,q=None,qReduced=None):
    '''
    Given a kinematic model and its reduced version, a geometry model corresponding
    to the unreduced model, the corresponding data and a reference configuration (in
    both unreduced and reduced version), modify the geometry model to make it corresponds
    to the reduced kinematic tree.
    '''
    if q is not None:
        pio.forwardKinematics(model,data,q)
        pio.updateGeometryPlacements(model,data,geomModel,geomData)
    if qReduced is not None:
        pio.forwardKinematics(modelReduced,dataReduced,qReduced)

    for igeom,geom in enumerate(geomModel.geometryObjects):
        jname = model.names[geom.parentJoint]
        if(modelReduced.existJointName(jname)):
            # Case 1: the parent joint just change ID, not name
            geom.parentJoint = modelReduced.getJointId(jname)
        else:
            # Case 2: Geom is now orphan'
            # Get joint ID from the parent frame
            frame = modelReduced.frames[modelReduced.getFrameId(model.frames[geom.parentFrame].name)]
            geom.parentJoint = frame.parent
            # Change geom placement wrt new joint
            oMg = geomData.oMg[igeom]
            oMj = dataReduced.oMi[geom.parentJoint]
            jMg = oMj.inverse()*oMg
            geom.placement = jMg

def loadSoloLeg(initViewer=False):
    # Load Solo8.
    URDF_FILENAME = "solo.urdf"
    SRDF_FILENAME = "solo.srdf"
    SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)
    robot = pio.RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath])
    readParamsFromSrdf(robot, modelPath + SRDF_SUBPATH, False, False, "standing")

    # Create Solo-leg model from Solo8
    lock = list(range(3,9))
    rmodel = pio.buildReducedModel(robot.model,lock,robot.q0)
    rdata  = rmodel.createData()

    # Create solo-leg geometry model (visuals and collisions
    reduceGeomModel(robot.model,robot.data,rmodel,rdata,robot.visual_model,robot.visual_data,robot.q0,robot.q0[:2])  
    reduceGeomModel(robot.model,robot.data,rmodel,rdata,robot.collision_model,robot.collision_data)  

    # Create Solo robot-wrapper
    rw=pio.RobotWrapper(rmodel,collision_model=robot.collision_model,visual_model=robot.visual_model)
    if initViewer:
        rw.initViewer(loadModel=True)
        for g in rw.visual_model.geometryObjects: 
            if g.parentJoint==0:
                rw.viewer.gui.setVisibility('world/pinocchio/visuals/'+g.name,'OFF')

    return rw

if __name__ == "__main__":
    import time
    robot = loadSoloLeg(initViewer=True)
    robot.display(robot.q0+1)
    for i in range(100): 
        robot.display(robot.q0+i/10.) 
        time.sleep(.1)        
