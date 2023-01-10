import unittest
import pybullet as p
import numpy as np
import os
from xml.dom.minidom import getDOMImplementation, Element
from creature import creature, genome

class CreatureLinkTest(unittest.TestCase):
    def testCreatureLinkClass(self):
        self.assertIsNotNone(creature.CreatureLink)
        self.assertIsNotNone(creature.CreatureLink("A", "None", 1))



class CreatureTest(unittest.TestCase):
    def testCreatureClass(self):
        self.assertIsNotNone(creature.Creature)

        spec = genome.GeneSpec.get_gene_spec()
        cr = creature.Creature(5, spec)
        self.assertIsNotNone(cr)
        self.assertEqual(cr.spec, spec)
        self.assertEqual(type(cr.dna), np.ndarray)
        self.assertEqual(cr.dna.shape, (5, len(spec)))
        
    def testCreatureGenomeToLinks(self):
        self.assertIsNotNone(creature.Creature.genome_to_links)

        spec = genome.GeneSpec.get_gene_spec()
        dna = genome.Genome.init_random_genome(5, len(spec))
        g_dicts = genome.Genome.genome_to_dict(dna, spec)

        links = creature.Creature.genome_to_links(g_dicts)
        self.assertIsNotNone(links)
        self.assertEqual(len(links), len(g_dicts))
        self.assertEqual(type(links[0]), creature.CreatureLink)
        self.assertEqual(links[0].parent_name, "None")
        self.assertEqual(links[0].recur, 1)
        self.assertEqual(links[0].name, links[1].parent_name)
        self.assertEqual(links[-2].name, links[-1].parent_name)

    def testCreatureFlatLinks(self):
        self.assertIsNotNone(creature.Creature.get_flat_links)

        spec = genome.GeneSpec.get_gene_spec()
        cr = creature.Creature(5, spec)
        self.assertIsNotNone(cr)
        
        flat_links = cr.get_flat_links()
        self.assertEqual(len(flat_links), 5)
        self.assertEqual(type(flat_links[0]), creature.CreatureLink)
        self.assertEqual(flat_links[0].parent_name, "None")
        self.assertEqual(flat_links[0].recur, 1)
        self.assertEqual(flat_links[0].name, flat_links[1].parent_name)
        self.assertEqual(flat_links[-2].name, flat_links[-1].parent_name)
        
    def testCreatureManualExtendedLinks(self):
        self.assertIsNotNone(creature.Creature.expand_links)

        spec = genome.GeneSpec.get_gene_spec()
        for _ in range(10):
            cr = creature.Creature(5, spec)
            flat_links = cr.get_flat_links()
            exp_links = creature.Creature.expand_links(flat_links)

            # manually count the number of links in expanded link list
            link_count = 1
            last_count = 1
            for i in range(1, len(flat_links)):
                last_count = last_count * flat_links[i].recur
                link_count += last_count

            self.assertIsNotNone(exp_links)
            self.assertEqual(type(exp_links), list)
            self.assertEqual(type(flat_links[0]), creature.CreatureLink)
            self.assertEqual(type(flat_links[-1]), creature.CreatureLink)
            self.assertEqual(type(exp_links[0]), creature.CreatureLink)
            self.assertEqual(type(exp_links[-1]), creature.CreatureLink)
            self.assertEqual(link_count, len(exp_links))
            self.assertGreaterEqual(len(exp_links), len(flat_links))


        spec = genome.GeneSpec.get_gene_spec()
        
    def testCreatureExpandedLinks(self):
        self.assertIsNotNone(creature.Creature.get_expanded_links)

        spec = genome.GeneSpec.get_gene_spec()
        cr = creature.Creature(5, spec)
        exp_links = cr.get_expanded_links()

        for _ in range(10):
            cr = creature.Creature(5, spec)
            flat_links = cr.get_flat_links()
            exp_links = cr.get_expanded_links()

            # manually count the number of links in expanded link list
            link_count = 1
            last_count = 1
            for i in range(1, len(flat_links)):
                last_count = last_count * flat_links[i].recur
                link_count += last_count

            self.assertIsNotNone(exp_links)
            self.assertEqual(type(exp_links), list)
            self.assertEqual(type(flat_links[0]), creature.CreatureLink)
            self.assertEqual(type(flat_links[-1]), creature.CreatureLink)
            self.assertEqual(type(exp_links[0]), creature.CreatureLink)
            self.assertEqual(type(exp_links[-1]), creature.CreatureLink)
            self.assertEqual(link_count, len(exp_links))
            self.assertGreaterEqual(len(exp_links), len(flat_links))



class CreatureXMLTest(unittest.TestCase):
    def testCreatureLinkToXML(self):
        spec = genome.GeneSpec.get_gene_spec()
        gene = genome.Genome.init_random_gene(len(spec))
        adom = getDOMImplementation().createDocument(None, "start", None)

        g_dict = genome.Genome.gene_to_dict(gene, spec)
        
        cr = creature.CreatureLink("A",  g_dict, "Parent", 1)
        link, joint = creature.Creature.link_to_xml(cr.name, cr.parent_name, cr.gene_dict, adom)
        self.assertIsInstance(link, Element)
        self.assertIsInstance(joint, Element)

    def testCreatureLinkXML(self):
        self.assertIsNotNone(creature.Creature.creature_to_xml)

        spec = genome.GeneSpec.get_gene_spec()
        for _ in range(10):
            cr = creature.Creature(5, spec)
            cr.get_flat_links()
            links = cr.get_expanded_links()

            robot_tag1 = creature.Creature.creature_to_xml(links)
            robot_tag2 = cr.get_robot_xml()

            self.assertIsNotNone(robot_tag1)
            self.assertIsNotNone(robot_tag2)
            self.assertIsInstance(robot_tag1, Element)
            self.assertIsInstance(robot_tag2, Element)
            self.assertEqual(robot_tag1.toxml(), robot_tag2.toxml())

            self.assertEqual(len(robot_tag1.childNodes), len(cr.get_expanded_links()) * 2 - 1)

    def testCreatureWriteXML(self):
        spec = genome.GeneSpec.get_gene_spec()
        cr = creature.Creature(5, spec)
        cr.write_robot_xml(".temp/test_motors.urdf")
        self.assertTrue(os.path.exists(".temp/test_motors.urdf"))

        with open(".temp/test_motors.urdf") as f:
            f_str= f.read()

        self.assertEqual(f_str, cr.get_robot_xml().toprettyxml())

    def testCreatureLoadXML(self):
        spec = genome.GeneSpec.get_gene_spec()

        for _ in range(5):
            c = creature.Creature(gene_count = 3, gene_spec = spec)
            xml_str = c.get_robot_xml().toprettyxml()
            with open('.temp/test.urdf', 'w') as f:
                f.write(xml_str)
            p.connect(p.DIRECT)
            cid = p.loadURDF('.temp/test.urdf')
            self.assertIsNotNone(cid)

        c = creature.Creature(gene_count = 5, gene_spec = spec)
        xml_str = c.get_robot_xml().toprettyxml()
        with open('.temp/test.urdf', 'w') as f:
            f.write(xml_str)
        p.connect(p.DIRECT)
        cid = p.loadURDF('.temp/test.urdf')
        self.assertIsNotNone(cid)

class CreatureMoveTest(unittest.TestCase):
    def testMovingDistance(self):
        self.assertIsNotNone(creature.Creature.update_position)
        self.assertIsNotNone(creature.Creature.get_distance)

        cr = creature.Creature(2)
        self.assertEqual(0, cr.get_distance())
        self.assertEqual(cr.start_position, (0, 0, 0))
        self.assertEqual(cr.last_position, (0, 0, 0))

        cr.update_position((0, 0, 1))
        self.assertGreater(cr.get_distance(), 0)
        self.assertEqual(cr.start_position, (0, 0, 0))
        self.assertEqual(cr.last_position, (0, 0, 1))