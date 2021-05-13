import torch
import torch.nn as nn
import numpy as np
import pdb
data = 50 + 25 * torch.randn(1000)

hist = torch.histc(data, bins=10, min=0, max=100)

print(hist)

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

softhist = SoftHistogram(bins=10, min=0, max=100, sigma=3)
pdb.set_trace()
data.requires_grad = True
hist = softhist(data)
print(hist)

hist.sum().backward()
print(data.grad.max())