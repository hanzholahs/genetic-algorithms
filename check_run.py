import time
import pybullet as p
import random
import pybullet_data
from ga_creature import creature


c = creature.Creature(gene_count=4)
xml_str = c.get_robot_xml().toprettyxml()
motors = c.get_motors()

with open('urdf/creature.urdf', 'w') as f:
    f.write(xml_str)


p.connect(p.GUI, options="--opengl2")

# configuration of the engine:
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane_shape = p.createCollisionShape(p.GEOM_PLANE)
floor = p.createMultiBody(plane_shape, plane_shape)

rob1 = p.loadURDF('urdf/creature.urdf')
step = 0

while True:
    assert len(motors) == p.getNumJoints(rob1), "bad motors!"
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