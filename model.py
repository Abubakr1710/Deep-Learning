from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network).__init__()
        self.conv1=nn.Conv2d(1,32,5, padding=1,stride=2)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,16,5)
        
        self.fc1=nn.Linear(16*1*1, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self,x):
        