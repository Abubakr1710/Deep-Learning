from pickle import TRUE
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(0)

def dataloader():

    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
    
    trainset=datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader=DataLoader(trainset, batch_size=32, shuffle=True)

    testset=datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)
    testloader=DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader