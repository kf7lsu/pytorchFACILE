import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantNet(nn.Module):
    def __init__(self, n_features):
        super(QuantNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_features, momentum=0.6)
        self.fc1 = nn.Linear(n_features, 12)
        self.drop1 = nn.Dropout(0.6)
        self.bn2 = nn.BatchNorm1d(12, momentum=0.2)
        self.fc2 = nn.Linear(12, 5)
        self.bn3 = nn.BatchNorm1d(5, momentum=0.6)
        self.fc3 = nn.Linear(5, 3)
        self.bn4 = nn.BatchNorm1d(3, momentum=0.6)
        self.fc4 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        x = F.relu(self.fc4(x))
        return x
