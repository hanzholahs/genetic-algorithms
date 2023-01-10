import os
import re
import numpy as np
import pybullet as p
import time
from simulation import population, simulation, manipulation

eval_fitness = manipulation.Selection.eval_fitness

# Simulation parameters
num_of_processes = 10
max_sim_frames = 2400
dir_path = os.path.dirname(os.path.realpath(__file__))
base_folder = os.path.join(dir_path, "data")
identifier  = "test_dna"

pop = population.Population(1, 1)
sim = simulation.MultiProcessSim(num_of_processes)

assert os.path.exists(base_folder) and len(os.listdir(base_folder)) > 0
gen_dirs  = [os.path.join(base_folder, d) for d in os.listdir(base_folder)]
latest_gen_dir = max(gen_dirs, key=os.path.getmtime)
last_iter = int(re.findall('[0-9]+$', latest_gen_dir)[0])
f = os.listdir(latest_gen_dir)[0]
pop.pop_from_csvs(latest_gen_dir, identifier)

sim.eval_population(pop)

fits = eval_fitness(pop.creatures)
fittest = pop.creatures[np.argmax(fits)]

xml_str = fittest.get_robot_xml().toprettyxml()
motors = fittest.get_motors()

with open('./fittest.urdf', 'w') as f:
    f.write(xml_str)


p.connect(p.GUI, options="--opengl2")

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