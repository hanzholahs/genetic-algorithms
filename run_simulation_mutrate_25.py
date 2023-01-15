import os
import re
from datetime import datetime
from simulation import population, simulation
from simulation.log import write_log_file

# Generation parameters
len_gene_init = 5
num_of_creatures = 100
num_of_elites = 5
num_of_random = 5
min_len = 2
max_len = 10
max_growth = 1.2
mut_amt  = .1
mut_rate = .25

# Simulation parameters
num_of_processes = 10
max_sim_frames = 2400
iter_count = 0
iter_stop  = 200

simulation_id = "sim-mutrate-25"
dir_path = os.path.dirname(os.path.realpath(__file__))
cr_dna_path = os.path.join(dir_path, f"data/{simulation_id}/dna")
cr_fit_path = os.path.join(dir_path, f"data/{simulation_id}/fittest_dna")
report_path = os.path.join(dir_path, f"data/{simulation_id}/report")
simlog_path = os.path.join(dir_path, f"data/{simulation_id}/log.txt")


pop = population.Population(num_of_creatures, len_gene_init)
sim = simulation.MultiProcessSim(num_of_processes)

if os.path.exists(cr_dna_path) and len(os.listdir(cr_dna_path)) > 0:
    gen_dirs  = [os.path.join(cr_dna_path, d) for d in os.listdir(cr_dna_path)]
    latest_gen_dir = max(gen_dirs, key=os.path.getmtime) # latest modified, based on crlf, 2020, https://stackoverflow.com/a/60113327
    last_iter = int(re.findall('[0-9]+$', latest_gen_dir)[0])
    iter_count += last_iter
    iter_stop += last_iter
    f = os.listdir(latest_gen_dir)[0]
    pop.pop_from_csvs(latest_gen_dir, simulation_id)
    num_of_creatures = len(pop.creatures)
else:
    os.makedirs(cr_dna_path, exist_ok = True)

time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
simlog_lines = [f"{time} Starting simulation..."]
print(simlog_lines[0])

for i in range(iter_count+1, iter_stop+1):
    sim.eval_population(pop, max_sim_frames)

    pop.generate_pop_report(i, report_path)
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

    time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    simlog_lines.append(f"{time} Iteration: {i}")
    pop.fittest_to_csvs(num_of_elites, f"{cr_fit_path}/{simulation_id}_iter_{i}", simulation_id)

    if (i) % 10 == 0:
        pop.pop_to_csvs(f"{cr_dna_path}/{simulation_id}_iter_{i}", simulation_id)
        print(simlog_lines[i - iter_count])

write_log_file(
    simlog_path, 
    simlog_lines,
    simulation_id = simulation_id,
    num_of_processes = num_of_processes,
    max_sim_frames = max_sim_frames,
    iter_count = iter_count,
    iter_stop  =iter_stop,
    num_of_creatures = num_of_creatures,
    num_of_elites = num_of_elites,
    num_of_random = num_of_random,
    min_len = min_len,
    max_len = max_len,
    max_growth = max_growth,
    mut_amt = mut_amt,
    mut_rate = mut_rate
)