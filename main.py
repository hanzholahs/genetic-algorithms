import os
from simulation import population, simulation

num_of_generations = 5
max_sim_frame = 2400
base_folder = "data"
identifier  = "dna"

pop = population.Population(10, 5)
sim = simulation.MultiProcessSim(10)

# if os.path.exists(base_folder) and len(os.listdir(base_folder)) > 0:
#     gen_dirs  = [d for d in os.listdir(base_folder) if os.path.isdir(d)]
#     latest_gen_dir = max(gen_dirs, key=os.path.getmtime)
#     pop.pop_from_csv(os.path.join(base_folder, latest_gen_dir), identifier)
# else:
#     os.makedirs(base_folder, exist_ok = True)

for ind_gen in range(1, num_of_generations + 1):
    print(f"Iteration: {ind_gen}")
    sim.eval_population(pop, max_sim_frame)
    pop.reset_population_new_gen(num_elites = 5, num_new_random = 5)
    pop.pop_to_csv(f"{base_folder}/iteration_{ind_gen}", "test_dna")