import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, resnet50

class Neural_network(nn.Module):
    def __init__(self):
        super(Neural_network, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 62)
        
    def forward(self, x):
        return self.model(x)