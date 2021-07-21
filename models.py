import torch
import torch.nn as nn

def mini_model():
    return nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 32, 1000),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 10))