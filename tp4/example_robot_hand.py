import pinocchio as pio
pio.switchToNumpyMatrix()
from robot_hand import RobotHand

robot = RobotHand()
robot.display(robot.q0)

