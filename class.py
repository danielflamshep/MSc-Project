
import numpy as np

class exp:
    def __init__(self):
        self.w = None

    def train(self, num):
        self.w = np.random.randn(num, num)