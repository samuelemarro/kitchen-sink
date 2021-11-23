import torch
import torch.nn as nn

def mini_model(coeff=1):
    return nn.Sequential(
            nn.Conv2d(3, int(8 * coeff), 3),
            nn.ReLU(),
            nn.Conv2d(int(8 * coeff), int(16 * coeff), 3),
            nn.ReLU(),
            nn.Conv2d(int(16 * coeff), int(32 * coeff), 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * int(32 * coeff), int(1000 * coeff)),
            nn.Linear(int(1000 * coeff), int(500 * coeff)),
            nn.ReLU(),
            nn.Linear(int(500 * coeff), 10))