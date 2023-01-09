import unittest
import numpy as np
from simulation import population, simulation, manipulation
from creature import creature

class PopulationClassTest(unittest.TestCase):
    def testPopulationClass(self):
        self.assertIsNotNone(population.Population)

        pop = population.Population(population_size = 10, gene_count = 3)
        self.assertIsNotNone(pop)
        self.assertEqual(len(pop.creatures), 10)
    
    def testRandomizedLinkLengthPopulation(self):
        self.assertIsNotNone(population.Population.reset_population_random_length)
        
        pop = population.Population(population_size = 20, gene_count = 10)
        pop.reset_population_random_length()
        for cr in pop.creatures:
            links = cr.get_flat_links()
            self.assertGreaterEqual(len(links), 2)
            self.assertLess(len(links), 10)

    def testResetPopulation(self):
        self.assertIsNotNone(population.Population.reset_population)

        pop = population.Population(20, 10)
        self.assertEqual(len(pop.creatures), 20)
        pop.reset_population([creature.Creature(10), creature.Creature(10)])
        self.assertEqual(len(pop.creatures), 2)
        with self.assertRaises(AssertionError):
            pop.reset_population(creature.Creature(10))

    def testResetPopulationManually(self):
        cr_1 = creature.Creature(15)
        cr_1.start_position = (0, 0, 0)
        cr_1.last_position  = (10, 100, 0)
        
        cr_2 = creature.Creature(5)
        cr_2.start_position = (0, 0, 0)
        cr_2.last_position  = (10, 100, 0)

        pop = population.Population(2)
        pop.reset_population([cr_1, cr_2])
        self.assertEqual(len(pop.creatures), 2)
        self.assertIsInstance(pop.creatures[0], creature.Creature)
        self.assertEqual(pop.creatures[0], cr_1)
        self.assertEqual(pop.creatures[1].last_position, cr_2.last_position)

    def testSelectParents(self):
        self.assertIsNotNone(population.Population.select_parents)

        pop_size = 5
        pop = population.Population(pop_size, 5)
        sim = simulation.MultiProcessSim(5)
        sim.eval_population(pop)
        
        parents = pop.select_parents()
        self.assertIsNotNone(parents)
        self.assertEqual(type(parents), tuple)
        self.assertEqual(len(parents), 2)
        self.assertIsInstance(parents[0], creature.Creature)
        self.assertIsInstance(parents[1], creature.Creature)

class PopulationCSVExportingTest(unittest.TestCase):
    def testCSVGeneration(self):
        for _ in range(10):
            pop = population.Population(10, 5)
            dna_1 = pop.creatures[0].dna
            dna_2 = pop.creatures[1].dna

            pop.pop_to_csv(base_folder = ".temp", identifier = "simulation")
            pop.pop_from_csv(base_folder = ".temp", identifier = "simulation")
            dna_3 = pop.creatures[0].dna
            dna_4 = pop.creatures[1].dna

            self.assertTrue(np.mean(dna_1 == dna_3) == 1)
            self.assertTrue(np.mean(dna_2 == dna_4) == 1)

class NewGenerationTest(unittest.TestCase):
    def testNewGenerationManually(self):
        pop = population.Population(10, 3)
        sim = simulation.MultiProcessSim(5)
        for _1 in range(25):
            sim.eval_population(pop)
            new_creatures = []

            for _2 in range(pop.population_size):
                p1, p2 = pop.select_parents()
                self.assertIsNotNone(p1)
                self.assertIsNotNone(p2)
                self.assertIsInstance(p1, creature.Creature)
                self.assertIsInstance(p2, creature.Creature)

                child_dna = manipulation.Crossover.crossover_dna(p1.dna,  p2.dna)
                self.assertIsNotNone(child_dna)
                self.assertEqual(len(child_dna.shape), 2)
                self.assertEqual(child_dna.shape[1], p1.dna.shape[1])
                self.assertIsInstance(child_dna, np.ndarray)

                child_dna = manipulation.Mutation.point_mutate(child_dna)
                child_dna = manipulation.Mutation.shrink_mutate(child_dna)
                child_dna = manipulation.Mutation.grow_mutate(child_dna)
                
                cr = creature.Creature(1)
                cr.update_dna(child_dna)
                new_creatures.append(cr)

            pop.reset_population(new_creatures)

    def testNewGeneration(self):
        self.assertIsNotNone(population.Population.reset_population_new_gen)
        pop = population.Population(5, 3)
        sim = simulation.MultiProcessSim(5)
        for _ in range(25):
            sim.eval_population(pop)
            fits = [cr.get_distance() for cr in pop.creatures]
            fits.sort(reverse=True)
            for fit in fits:
                self.assertEqual(type(float(fit)), float)
            pop.reset_population_new_gen(2)
