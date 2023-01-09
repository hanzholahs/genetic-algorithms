import unittest
from simulation import simulation, population
from creature import creature

class SimulationClassTest(unittest.TestCase):
    def testSimulationClass(self):
        self.assertIsNotNone(simulation.Simulation)
        
        sim = simulation.Simulation()
        self.assertIsNotNone(sim)
        self.assertIsNotNone(sim.client_id)

    def testRunACreatureInSimulation(self):
        self.assertIsNotNone(simulation.Simulation.run_creature)

        sim = simulation.Simulation()
        cr  = creature.Creature(3)
        sim.run_creature(cr)
        self.assertNotEqual(cr.start_position, cr.last_position)
        self.assertGreater(cr.get_distance(), 0)

    def testSimulationRunForPopulation(self):
        sim = simulation.Simulation()
        pop = population.Population(5, 4)

        for cr in pop.creatures:
            sim.run_creature(cr)
        
        dists = [cr.get_distance() for cr in pop.creatures]
        self.assertEqual(len(dists), 5)
        for dist in dists:
            self.assertGreater(dist, 0)

    def testSerializedPopulationRun(self):
        pop_size = 5
        sim = simulation.Simulation()
        pop = population.Population(pop_size, 4)
        sim.eval_population(pop)
        
        dists = [cr.get_distance() for cr in pop.creatures]
        self.assertEqual(len(dists), pop_size)
        for dist in dists:
            self.assertGreater(dist, 0)

    def testMultiProcessPopRun(self):
        self.assertIsNotNone(simulation.MultiProcessSim)
        pop_size = 5
        pop = population.Population(pop_size, 5)
        multisim = simulation.MultiProcessSim(5)
        multisim.eval_population(pop)
        
        dists = [cr.get_distance() for cr in pop.creatures]
        self.assertEqual(len(dists), pop_size)
        for dist in dists:
            self.assertGreater(dist, 0)