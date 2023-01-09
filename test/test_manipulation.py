import unittest
import copy
import numpy as np
from creature import creature, genome
from simulation import manipulation, simulation, population

class SelectionClassTest(unittest.TestCase):
    def testSelectionClass(self):
        self.assertIsNotNone(manipulation.Selection)
    
    def testFitnessFunction(self):
        pop_size = 5
        pop = population.Population(pop_size, 5)
        multisim = simulation.MultiProcessSim(5)
        multisim.eval_population(pop)
        
        fits = manipulation.Selection.eval_fitness(pop.creatures)
        self.assertEqual(len(fits), pop_size)
        for fit in fits:
            self.assertGreaterEqual(fit, 0)
    
    def testFitnessEvaluationScenarioDiffLength(self):
        cr_1 = creature.Creature(10)
        cr_1.start_position = (0, 0, 0)
        cr_1.last_position  = (10, 100, 0)
        
        cr_2 = creature.Creature(5)
        cr_2.start_position = (0, 0, 0)
        cr_2.last_position  = (10, 100, 0)

        pop = population.Population(2)
        pop.reset_population([cr_1, cr_2])

        fits = manipulation.Selection.eval_fitness(pop.creatures)
        self.assertLess(fits[0], fits[1])
    
    def testFitnessEvaluationScenarioDiffDist(self):
        cr_1 = creature.Creature(5)
        cr_1.last_position  = (100, 100, 0)

        cr_2 = copy.copy(cr_1) 
        cr_2.last_position  = (100, 120, 0)
        cr_3 = copy.copy(cr_1) 
        cr_3.last_position  = (120, 100, 0)
        cr_4 = copy.copy(cr_1) 
        cr_4.last_position  = (120, 120, 0)

        pop = population.Population(2)
        pop.reset_population([cr_1, cr_2, cr_3, cr_4])

        fits = manipulation.Selection.eval_fitness(pop.creatures)
        self.assertLess(fits[0], fits[1])
        self.assertLess(fits[0], fits[2])
        self.assertLess(fits[1], fits[2])



    def testParentSelection(self):
        self.assertIsNotNone(manipulation.Selection.select_parent_indices)
        self.assertIsNotNone(manipulation.Selection.select_parents)
        fits = [1, 1, 1, 2, 5]
        parents = ["A", "B", "C", "D", "E"]
        indices = manipulation.Selection.select_parent_indices(fits)

        for _ in range(5):
            indices = manipulation.Selection.select_parent_indices(fits)
            p1, p2 = parents[indices[0]], parents[indices[1]]
            
            self.assertTrue(np.issubdtype(indices[0], int))
            self.assertTrue(np.issubdtype(indices[1], int))
            self.assertIn(p1, parents)
            self.assertIn(p2, parents)
            self.assertNotEqual(p1, p2)
            
            p1, p2 = manipulation.Selection.select_parents(parents, fits)
            self.assertIn(p1, parents)
            self.assertIn(p2, parents)
            self.assertNotEqual(p1, p2)

    def testParentSelectionBasedOnSimulation(self):
        pop_size = 5
        pop = population.Population(pop_size, 5)
        multisim = simulation.MultiProcessSim(5)
        multisim.eval_population(pop)
        
        fits = manipulation.Selection.eval_fitness(pop.creatures)
        p1, p2 = manipulation.Selection.select_parents(pop.creatures, fits)
        self.assertIsInstance(p1, creature.Creature)
        self.assertIsInstance(p2, creature.Creature)
        self.assertIn(p1, pop.creatures)
        self.assertIn(p2, pop.creatures)
        self.assertEqual(len(p1.get_flat_links()), 5)
        self.assertEqual(len(p2.get_flat_links()), 5)

class CrossoverClassTest(unittest.TestCase):
    def testCrossoverClass(self):
        self.assertIsNotNone(manipulation.Crossover)

    def testCrossoverGenes(self):
        gene_1 = genome.Genome.init_random_gene(10)
        gene_2 = genome.Genome.init_random_gene(10)
        for _ in range(15):
            child_gene = manipulation.Crossover.crossover_dna(gene_1, gene_2)
            self.assertIsNotNone(child_gene)
            self.assertGreater(len(child_gene), 1)
            self.assertLess(len(child_gene), len(gene_1) + len(gene_2))
        
        genome_1 = genome.Genome.init_random_genome(10, 10)
        genome_2 = genome.Genome.init_random_genome(10, 10)
        for _ in range(15):
            child_genome = manipulation.Crossover.crossover_dna(genome_1, genome_2)
            self.assertIsNotNone(child_genome)
            self.assertGreater(len(child_genome), 1)
            self.assertLess(len(child_genome), len(genome_1) + len(genome_2))
        
class MutationClassTest(unittest.TestCase):
    def testMutationClass(self):
        self.assertIsNotNone(manipulation.Mutation)
    
    def testPointMutation(self):
        self.assertIsNotNone(manipulation.Mutation.point_mutate)

        dna = genome.Genome.init_random_genome(10, 25)
        rate = 0.25
        for _ in range(15):
            mutated_dna = manipulation.Mutation.point_mutate(dna, rate)
            self.assertEqual(np.mean(mutated_dna <= 1), 1)
            self.assertEqual(np.mean(mutated_dna >= 0), 1)
            self.assertEqual(dna.shape, mutated_dna.shape)
            self.assertTrue(np.mean(dna == mutated_dna) < (1 - rate + 3 * rate))
            self.assertTrue(np.mean(dna == mutated_dna) > (1 - rate - 3 * rate))

        mutated_dna = manipulation.Mutation.point_mutate(dna, 0)
        self.assertTrue(np.mean(dna == mutated_dna) == 1)

        mutated_dna = manipulation.Mutation.point_mutate(dna, 1)
        self.assertTrue(np.mean(dna != mutated_dna) > .95)
        
    def testShrinkMutation(self):
        self.assertIsNotNone(manipulation.Mutation.shrink_mutate)

        dna = genome.Genome.init_random_genome(25, 5)
        rate = 0.25
        for _ in range(15):
            mutated_dna = manipulation.Mutation.shrink_mutate(dna, rate)
            self.assertLessEqual(len(mutated_dna), len(dna))
            self.assertEqual(mutated_dna.shape[1], dna.shape[1])
            self.assertTrue(len(mutated_dna) / len(dna) < (1 - rate + 3 * rate))
            self.assertTrue(len(mutated_dna) / len(dna) > (1 - rate - 3 * rate))

        mutated_dna = manipulation.Mutation.shrink_mutate(dna, 1)
        self.assertTrue(len(mutated_dna) == 0)
        
        mutated_dna = manipulation.Mutation.shrink_mutate(dna, 0)
        self.assertTrue(len(mutated_dna) == len(dna))
        self.assertTrue(np.mean(dna == mutated_dna) == 1)\
                    
    def testGrowMutation(self):
        self.assertIsNotNone(manipulation.Mutation.grow_mutate)

        dna = genome.Genome.init_random_genome(25, 5)
        rate = 0.25
        for _ in range(15):
            mutated_dna = manipulation.Mutation.grow_mutate(dna, rate)
            self.assertGreaterEqual(len(mutated_dna), len(dna))
            self.assertEqual(mutated_dna.shape[1], dna.shape[1])
            self.assertTrue(len(mutated_dna) / len(dna) < (1 + rate + 3 * rate))
            self.assertTrue(len(mutated_dna) / len(dna) > (1 + rate - 3 * rate))

        mutated_dna = manipulation.Mutation.grow_mutate(dna, 1)
        self.assertTrue(len(mutated_dna) == 2 * len(dna))
        self.assertEqual(mutated_dna.shape[1], dna.shape[1])
        self.assertEqual(np.mean(mutated_dna[0] == mutated_dna[len(dna)]), 1)
        
        mutated_dna = manipulation.Mutation.grow_mutate(dna, 0)
        self.assertTrue(len(mutated_dna) == len(dna))
        self.assertEqual(mutated_dna.shape[1], dna.shape[1])
        self.assertTrue(np.mean(dna == mutated_dna) == 1)