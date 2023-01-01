import unittest
import numpy as np
from ga_creature import genome

class GenomeTest(unittest.TestCase):
    def testGenomeClass(self):
        self.assertIsNotNone(genome.Genome)

    def testGenomeRandomGeneInitializer(self):
        self.assertIsNotNone(genome.Genome.init_random_gene)

        for _ in range(10):
            gene_length = np.random.randint(5, 10)
            gene = genome.Genome.init_random_gene(gene_length)
            self.assertEqual(type(gene), np.ndarray)
            self.assertEqual(len(gene), gene_length)
            
            for i in range(gene_length):
                self.assertGreaterEqual(gene[i], 0)
                self.assertLessEqual(gene[i], 1)

    def testGenomeRandomGenomeInitializer(self):
        self.assertIsNotNone(genome.Genome.init_random_genome)

        data = genome.Genome.init_random_genome(20, 5)
        self.assertIsNotNone(data)
        self.assertIsNotNone(data[0])
        self.assertIsNotNone(data[0][0])
        self.assertEqual(type(data), np.ndarray)
        self.assertEqual(data.shape, (20, 5))
        self.assertEqual(len(data), 20)
        self.assertEqual(len(data[0]), 5)

    def testGenomeGeneToDict(self):
        self.assertIsNotNone(genome.Genome.gene_to_dict)

        spec = genome.GeneSpec.get_gene_spec()
        gene = genome.Genome.init_random_gene(len(spec))

        gene_dict = genome.Genome.gene_to_dict(gene, spec)
        self.assertEqual(type(gene_dict), dict)
        self.assertEqual(len(gene_dict), len(spec))
        self.assertIn("link_shape", gene_dict.keys())
        self.assertIn("link_recurrence", gene_dict.keys())
        self.assertIn("joint_axis_xyz", gene_dict.keys())
        self.assertIn("control_waveform", gene_dict.keys())
        self.assertEqual(type(gene_dict["link_shape"]), int)
        self.assertEqual(type(gene_dict["link_recurrence"]), int)
        self.assertEqual(type(gene_dict["joint_type"]), int)
        self.assertEqual(type(gene_dict["control_waveform"]), int)

        counter = []
        for _ in range(10):
            gene = genome.Genome.init_random_gene(len(spec))
            gene_dict = genome.Genome.gene_to_dict(gene, spec)
            if gene_dict["joint_origin_rpy_1"] > 1:
                counter.append(True)

        self.assertGreaterEqual(len(counter), 3)

    def testGenomeGenomeToDict(self):
        self.assertIsNotNone(genome.Genome.genome_to_dict)

        spec = genome.GeneSpec.get_gene_spec()
        data = genome.Genome.init_random_genome(20, len(spec))

        genome_dicts = genome.Genome.genome_to_dict(data, spec)
        self.assertEqual(type(genome_dicts), list)
        self.assertEqual(len(genome_dicts), 20)

        for i in range(len(genome_dicts)):
            self.assertEqual(type(genome_dicts[i]), dict)
            self.assertEqual(len(genome_dicts[i]), len(spec))
            
            for key in genome_dicts[i].keys():
                self.assertIn(key, spec.keys())



class GeneSpecTest(unittest.TestCase):
    def testGeneSpecClass(self):
        self.assertIsNotNone(genome.GeneSpec)

    def testGeneSpecDefinition(self):
        self.assertIsNotNone(genome.GeneSpec.get_gene_spec)

        spec = genome.GeneSpec.get_gene_spec()
        self.assertIsNotNone(spec)
        self.assertEqual(type(spec), dict)

        for key in spec.keys():
            self.assertEqual(type(spec[key]), dict)
            self.assertIn("scale", spec[key].keys())
            self.assertIn("type", spec[key].keys())
            self.assertIn("index", spec[key].keys())
            self.assertIn(spec[key]["type"], ["discrete", "continuous", "categorical"])

        self.assertEqual(spec["link_shape"]["type"], "categorical")
        self.assertEqual(spec["link_recurrence"]["type"], "discrete")
        self.assertEqual(spec["joint_origin_xyz_3"]["type"], "continuous")

    def testGeneSpecRedefinition(self):        
        new_spec = {
            "a": {"scale":1, "type":"categorical", "index":1}, 
            "b": {"scale":1, "type":"continuous", "index":2}, 
            "c": {"scale":10, "type":"discrete", "index":3}
        }
        self.assertTrue(genome.GeneSpec.valid_spec(new_spec))
        
        genome.GeneSpec.set_gene_spec(new_spec)
        spec = genome.GeneSpec.get_gene_spec()
        self.assertEqual(spec, new_spec)

        genome.GeneSpec.set_default_gene_spec()

    def testGeneSpecRedefinitionError(self):
        with self.assertRaises(AssertionError):
            new_spec = {"a":1, "b":2}
            genome.GeneSpec.set_gene_spec(new_spec)
        
        with self.assertRaises(AssertionError):
            new_spec = {"link": {"scale":1, "type": "", "index":1}}
            genome.GeneSpec.set_gene_spec(new_spec)
        
        with self.assertRaises(AssertionError):
            new_spec = {"link": {"scale":1, "index":1}}
            genome.GeneSpec.set_gene_spec(new_spec)
        
        with self.assertRaises(AssertionError):
            new_spec = {
                "link_shape": {"scale":1, "type": "discrete", "index":1},
                "link_length" : {"scale":1, "type": "discrete", "index":1}
            }
            genome.GeneSpec.set_gene_spec(new_spec)
        
        with self.assertRaises(AssertionError):
            new_spec = {"link": {"scale":"1.2", "type": "categorical", "index":1}}
            genome.GeneSpec.set_gene_spec(new_spec)
            
        genome.GeneSpec.set_default_gene_spec()
