import unittest
import time
import os
import pybullet as p
import numpy as np
from creature import genome, creature, motor

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

    def testDistanceMoved(self):
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        plane_shape = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(plane_shape, plane_shape)

        spec = genome.GeneSpec.get_gene_spec()
        cr = creature.Creature(5, spec)
        cr.write_robot_xml(".temp/test_motors.urdf")
        motors = cr.get_motors()
        robot  = p.loadURDF('.temp/test_motors.urdf')
        self.assertEqual(len(motors), p.getNumJoints(robot))

        start, _ = p.getBasePositionAndOrientation(robot)

        for i in range(2400):
            if i % 240 == 0:
                for joint_id, joint_motor in enumerate(motors):
                    p.setJointMotorControl2(
                        robot, 
                        joint_id, 
                        controlMode = p.VELOCITY_CONTROL, 
                        targetVelocity = joint_motor()
                    )
            p.stepSimulation()

            end, _ = p.getBasePositionAndOrientation(robot)
            
            dist_moved = np.linalg.norm(np.asarray(start) - np.asarray(end))
            self.assertGreaterEqual(dist_moved, 0)
