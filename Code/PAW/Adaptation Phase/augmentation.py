import random
from braindecode.augmentation import GaussianNoise
from torch import nn
from config import *

class DCShift(nn.Module):
    def __init__(self, probability=0.5, min=-10, max=10):
        super(DCShift, self).__init__()
        self.prob_threshold=probability
        self.min=min
        self.max=max
        
    def forward(self, data):
        if random.random()<self.prob_threshold:
            data = data + random.uniform(self.min, self.max)
        return data

transform_gaussian=GaussianNoise(probability=0.5, std=random.uniform(0, 0.2), random_state=args.seed)
transform_DCShift=DCShift()

def weak_aug():
    return nn.Sequential(
        transform_gaussian,
        transform_DCShift
    )
