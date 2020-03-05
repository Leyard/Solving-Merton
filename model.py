import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, 3)

    def forward(self, x):
        return x

    def num_flat_features(self, x):
        return x*x
