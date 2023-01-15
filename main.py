import time
import pybullet as p
import numpy as np
from creature import creature

dna = np.genfromtxt("./fittest_dna.csv", delimiter = ",")
cr = creature.Creature(1)
cr.update_dna(dna)

xml_str = cr.get_robot_xml().toprettyxml()
motors  = cr.get_motors()

with open('./fittest.urdf', 'w') as f:
    f.write(xml_str)

p.connect(p.GUI)

# configuration of the engine:
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

plane_shape = p.createCollisionShape(p.GEOM_PLANE)
floor = p.createMultiBody(plane_shape, plane_shape)

rob1 = p.loadURDF('./fittest.urdf')
step = 0
assert len(motors) == p.getNumJoints(rob1), "bad motors!"

while True:
    step += 1
    if step % 120 == 0:
        for jid, motor in enumerate(motors):
            vel = motor()
            p.setJointMotorControl2(
                rob1, 
                jid, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=vel)

    p.stepSimulation()
    time.sleep(1./240.)