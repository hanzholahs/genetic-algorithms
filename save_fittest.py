import os
import re
import numpy as np
import pybullet as p
import time
from simulation import population, simulation, manipulation

eval_fitness = manipulation.Selection.eval_fitness

# Set simulation parameters
num_of_processes = 10
max_sim_frames = 2400

# Find latest generation
simulation_id = "sim-final"
dir_path = os.path.dirname(os.path.realpath(__file__))
cr_dna_path = os.path.join(dir_path, f"data/{simulation_id}/dna")
cr_fit_path = os.path.join(dir_path, f"data/{simulation_id}/fittest_dna")

gen_dirs  = [os.path.join(cr_dna_path, d) for d in os.listdir(cr_dna_path)]
latest_gen_dir = max(gen_dirs, key=os.path.getmtime) # latest modified, based on crlf, 2020, https://stackoverflow.com/a/60113327
f = os.listdir(latest_gen_dir)[0]

# Load latest generation
pop = population.Population(1, 1)
sim = simulation.MultiProcessSim(20)
pop.pop_from_csvs(latest_gen_dir, simulation_id)

# Evaluate population
sim.eval_population(pop)

# Save fittest creatures based on distance
fits = [cr.get_distance() for cr in pop.creatures]
fittest = pop.creatures[np.argmax(fits)]

np.savetxt("./fittest_dna.csv", fittest.dna, delimiter = ",")

