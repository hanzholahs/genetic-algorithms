import numpy as np
from enum import Enum

class MotorType(Enum):
    PULSE = 0
    SINE  = 1

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        self.motor_type = MotorType(control_waveform)
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0
        self.motor_function = self.get_motor_function()        

    def __call__(self):
        self.phase += self.freq
        return self.amp * self.motor_function(self.phase)
    
    def __repr__(self):
        return f"Motor\nType\t: {self.motor_type}\nAmp\t: {self.amp}\nFreq\t: {self.freq}\n"

    def get_motor_function(self):
        if self.motor_type == MotorType.PULSE:
            def motor_function(phase):
                return ((-1) ** (np.ceil(phase % (np.pi * 2)) + 1))
        elif self.motor_type == MotorType.SINE:
            def motor_function(phase):
                return np.sin(phase)
        self.motor_function = motor_function
        return self.motor_function