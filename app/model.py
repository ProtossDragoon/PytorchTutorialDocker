# 서드파티
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 64, 5, padding=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(64, 64, 1)
        self.fc2 = nn.Conv2d(64, 32, 1)
        self.fc3 = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = F.softmax(x, dim=-1)
        return x
