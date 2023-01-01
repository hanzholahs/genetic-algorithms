import unittest
import numpy as np
from ga_creature import genome, creature, motor

class MotorClassTest(unittest.TestCase):
    def testMotorClass(self):
        self.assertIsNotNone(motor.Motor)
        self.assertIsNotNone(motor.Motor(1, 50, 50))

    def testMotorOutput(self):
        m = motor.Motor(0, 2, .5)
        self.assertEqual(m.motor_type, motor.MotorType(0))
        self.assertEqual(m(), 2)
        self.assertEqual(m(), 2)
        self.assertEqual(m(), -2)
        self.assertEqual(m(), -2)

        m = motor.Motor(1, 2, np.pi / 2 - .001)
        self.assertEqual(m.motor_type, motor.MotorType(1))
        self.assertLessEqual(m(), 2)
        self.assertLessEqual(m(), 2)
        self.assertLessEqual(m(), 0)
        self.assertLessEqual(m(), 0)

class MotorCreatureTest(unittest.TestCase):
    def testCreatureMotor(self):
        self.assertIsNotNone(creature.Creature.get_motors)
        cr = creature.Creature(3)
        cr.get_expanded_links()
        
        motors = cr.get_motors()
        self.assertIsNotNone(motors)
        for motor in motors:
            self.assertLessEqual(motor(), 1)
            self.assertLessEqual(motor(), 1)
            self.assertLessEqual(motor(), 1)