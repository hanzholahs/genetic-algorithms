import os
import re
from datetime import datetime
from simulation import population, simulation

# Generation parameters
len_gene_init = 5
num_of_creatures = 10
num_of_elites = 5
num_of_random = 5
min_len = 2
max_len = 15
max_growth = 1.2
mut_amt = .1
mut_rate = .1

# Simulation parameters
num_of_processes = 10
max_sim_frames = 2400
iter_count = 0
iter_stop  = 5
dir_path = os.path.dirname(os.path.realpath(__file__))
base_folder = os.path.join(dir_path, "data")
identifier  = "test_dna"

pop = population.Population(num_of_creatures, len_gene_init)
sim = simulation.MultiProcessSim(num_of_processes)

if os.path.exists(base_folder) and len(os.listdir(base_folder)) > 0:
    gen_dirs  = [os.path.join(base_folder, d) for d in os.listdir(base_folder)]
    latest_gen_dir = max(gen_dirs, key=os.path.getmtime)
    last_iter = int(re.findall('[0-9]+$', latest_gen_dir)[0])
    iter_count += last_iter
    iter_stop += last_iter
    f = os.listdir(latest_gen_dir)[0]
    pop.pop_from_csvs(latest_gen_dir, identifier)
else:
    os.makedirs(base_folder, exist_ok = True)

for i in range(iter_count+1, iter_stop+1):
    sim.eval_population(pop, max_sim_frames)
    pop.reset_population_new_gen(
        num_elites = num_of_elites,
        num_new_random = num_of_random,
        min_length_limit = min_len, 
        max_length_limit = max_len, 
        max_growth_rate = max_growth,
        point_mutation_amount = mut_amt,
        point_mutation_rate = mut_rate,
        shrink_mutation_rate = mut_rate,
        grow_mutation_rate = mut_rate
    )
    if (i+1) % 1 == 0:
        time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        print(f"{time} Iteration: {i}")
        pop.pop_to_csvs(f"{base_folder}/{identifier}_iter_{i}", identifier)