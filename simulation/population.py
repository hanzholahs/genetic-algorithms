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
            assert type(creatures) == list and type(creatures[0]) == creature.Creature
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

    def reset_new_generation(self, num_elite_creatures = 3):
        if num_elite_creatures < self.population_size :
            num_elite_creatures = 0

        fits = manipulation.Selection.eval_fitness(self.creatures)
        fittest_indices = fits.argsort()[-1:-4:-1]

        new_creatures = []

        for index in fittest_indices:
            new_creatures.append(self.creatures[index])

        for _ in range(self.population_size - num_elite_creatures):
            p1, p2 = self.select_parents()
            child_dna = manipulation.NewGeneration.generate_child_dna(p1.dna, p2.dna)
            child_cr = creature.Creature(1)
            child_cr.update_dna(child_dna)
            new_creatures.append(child_cr)

        self.reset_population(new_creatures)   

    def pop_to_csv(self, base_folder = ".", identifier = "dna"):
        Population.to_csv(self.creatures, base_folder = base_folder, identifier = identifier)

    def pop_from_csv(self, base_folder = ".", identifier = "dna"):
        new_creatures = Population.from_csv(base_folder = base_folder, identifier = identifier)
        self.reset_population(new_creatures)

    @staticmethod
    def to_csv(creatures, base_folder = ".", identifier = "dna"):
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
        for i, cr in enumerate(creatures):
            np.savetxt(f"{base_folder}/{identifier}_cr_{i}.csv", cr.dna, delimiter = ",")

    @staticmethod
    def from_csv(base_folder = ".", identifier = "dna"):
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

    