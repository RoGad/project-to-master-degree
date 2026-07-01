import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,4,2), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,128,4,2), nn.ReLU(),
            nn.Conv2d(128,256,4,2), nn.ReLU(),
            nn.Flatten(),
        )
        self.embed_dim = 256 * 2 * 2

    def forward(self,x):
        return self.net(x)