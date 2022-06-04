import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.random.seed(0)

def transformer():
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
    return transform

