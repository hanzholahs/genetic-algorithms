import os
import re
import numpy as np
from creature import creature, genome
from simulation import manipulation

class Population:
    def __init__(self, population_size, gene_count = 5):
        self.gene_count = gene_count
        self.population_size = population_size
        self.creatures = []
        self.best_score = 0

        self.reset_population()

    def add_creature(self, cr):
        self.creatures.append(cr)

    def reset_population(self, creatures = None):
        if creatures == None:
            self.creatures = [creature.Creature(self.gene_count) for _ in range(self.population_size)]
        else:
            assert type(creatures) == list and len(creatures) > 0
            assert type(creatures[0]) == creature.Creature
            for old_creature in self.creatures:
                del old_creature
            self.creatures = creatures

    def reset_population_random_length(self):
        assert self.gene_count >= 3
        new_creatures = []
        for _ in range(self.population_size):
            new_creatures.append(creature.Creature(np.random.randint(2, self.gene_count)))
        self.reset_population(new_creatures)

    def select_parents(self):
        fits = manipulation.Selection.eval_fitness(self.creatures)
        return manipulation.Selection.select_parents(self.creatures, fits)

    def reset_population_new_gen(
            self, 
            num_elites = 0,
            num_new_random = 0,
            min_length_limit = 2, 
            max_length_limit = 15, 
            max_growth_rate = 1.2,
            point_mutation_rate = .05,
            point_mutation_amount = .05,
            shrink_mutation_rate = .05,
            grow_mutation_rate = .05
        ):
        if num_elites < self.population_size :
            num_elites = 0
        if num_new_random > self.population_size:
            num_new_random = self.population_size - 1
        if num_elites + num_new_random > self.population_size:
            num_elites = int(self.population_size / 2)
            num_new_random = np.maximum(0, self.population_size - num_elites)
        
        fits = manipulation.Selection.eval_fitness(self.creatures)
        fittest_indices = np.array(fits).argsort()[-1:-(num_elites+1):-1] # find n argmax, based on NPE, 2011, https://stackoverflow.com/a/6910672

        new_creatures = []
        for index in fittest_indices:
            fittest_cr = self.creatures[index]
            new_creatures.append(fittest_cr)
        for _ in range(self.population_size - num_elites - num_new_random):
            p1, p2 = self.select_parents()
            child_dna = manipulation.NewGeneration.generate_child_dna(
                p1.dna, 
                p2.dna,
                min_length_limit,
                max_length_limit,
                max_growth_rate,
                point_mutation_rate,
                point_mutation_amount,
                shrink_mutation_rate,
                grow_mutation_rate
            )
            child_cr = creature.Creature(1)
            child_cr.update_dna(child_dna)
            new_creatures.append(child_cr)
        for _ in range(num_new_random):
            random_cr = creature.Creature(self.gene_count)
            new_creatures.append(random_cr)
        assert self.population_size == len(new_creatures)

        self.reset_population(new_creatures)   

    def pop_to_csvs(self, base_folder = ".", identifier = "dna"):
        Population.__to_csvs(self.creatures, base_folder = base_folder, identifier = identifier)

    def pop_from_csvs(self, base_folder = ".", identifier = "dna"):
        new_creatures = Population.__from_csvs(base_folder = base_folder, identifier = identifier)
        self.reset_population(new_creatures)

    def get_fittest_creatures(self, n_fittest = 3):
        fits = manipulation.Selection.eval_fitness(self.creatures)
        fittest_ids = fits.argsort()[-n_fittest:][::-1] # find n argmax, based on NPE, 2011, https://stackoverflow.com/a/6910672
        fittest_crs = [self.creatures[id] for id in fittest_ids]
        return fittest_crs

    def fittest_to_csvs(self, n_fittest = 3, base_folder = ".", identifier = "dna"):
        fittest_crs = self.get_fittest_creatures(n_fittest)
        Population.__to_csvs(fittest_crs, base_folder = base_folder, identifier = identifier)

    @staticmethod
    def __to_csvs(creatures, base_folder = ".", identifier = "dna"):
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        for i, cr in enumerate(creatures):
            np.savetxt(f"{base_folder}/{identifier}_cr_{i}.csv", cr.dna, delimiter = ",")

    @staticmethod
    def __from_csvs(base_folder = ".", identifier = "dna"):
        assert os.path.exists(base_folder)
        files = [f for f in os.listdir(base_folder) if re.match(f"^({identifier}).*\\.csv$", f)]
        files.sort()
        new_creatures = []
        for file in files:
            path = os.path.join(base_folder, file)
            dna = np.genfromtxt(path, delimiter = ",")
            cr = creature.Creature(1)
            cr.update_dna(dna)
            new_creatures.append(cr)
        return new_creatures

    def generate_pop_report(self, generation, base_folder = "."):
        n_exp_link  = [len(cr.get_expanded_links()) for cr in self.creatures]
        n_flat_link = [len(cr.get_flat_links()) for cr in self.creatures]
        dists = [cr.get_distance() for cr in self.creatures]
        fits  = list(manipulation.Selection.eval_fitness(self.creatures))

        file_names = [
            f"{generation}_n_exp_links.csv",
            f"{generation}_n_flat_links.csv",
            f"{generation}_distances.csv",
            f"{generation}_fitness.csv",
        ]

        Population.__generate_report_csv(file_names[0], n_exp_link, base_folder)
        Population.__generate_report_csv(file_names[1], n_flat_link, base_folder)
        Population.__generate_report_csv(file_names[2], dists, base_folder)
        Population.__generate_report_csv(file_names[3], fits, base_folder)

    @staticmethod
    def __generate_report_csv(csv_file_name, csv_rows, base_folder = "."):
        assert type(csv_rows) == list
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        with open(os.path.join(base_folder, csv_file_name), "w") as f:
            f.write(','.join(map(str, csv_rows)) + "\n")