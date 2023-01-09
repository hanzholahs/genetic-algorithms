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

    def __call__(self):
        self.phase += self.freq
        if self.motor_type == MotorType.PULSE:
            return self.amp * ((-1) ** (np.ceil(self.phase % (np.pi * 2)) + 1))
        elif self.motor_type == MotorType.SINE:
            return self.amp * np.sin(self.phase) 

    
    def __repr__(self):
        return f"Motor\nType\t: {self.motor_type}\nAmp\t: {self.amp}\nFreq\t: {self.freq}\n"