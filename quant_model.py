import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantLinear, QuantReLU
from constants import WEIGHT_BW, ACT_BW

class QuantNet(nn.Module):
    def __init__(self, n_features):
        super(QuantNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_features, momentum=0.6)
        #self.fc1 = nn.Linear(n_features, 12)
        self.fc1 = QuantLinear(n_features, 12, bias=True, weight_bit_width=WEIGHT_BW)
        self.relu1 = QuantReLU(bit_width=ACT_BW)
        self.drop1 = nn.Dropout(0.6)
        self.bn2 = nn.BatchNorm1d(12, momentum=0.2)
        #self.fc2 = nn.Linear(12, 5)
        self.fc2 = QuantLinear(12, 5, bias=True, weight_bit_width=WEIGHT_BW)
        self.relu2 = QuantReLU(bit_width=ACT_BW)
        self.bn3 = nn.BatchNorm1d(5, momentum=0.6)
        #self.fc3 = nn.Linear(5, 3)
        self.fc3 = QuantLinear(5, 3, bias=True, weight_bit_width=WEIGHT_BW)
        self.relu3 = QuantReLU(bit_width=ACT_BW)
        self.bn4 = nn.BatchNorm1d(3, momentum=0.6)
        #self.fc4 = nn.Linear(3, 1)
        self.fc4 = QuantLinear(3, 1, bias=True, weight_bit_width=WEIGHT_BW)
        self.relu4 = QuantReLU(bit_width=ACT_BW)

    def forward(self, x):
        x = self.bn1(x)
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.bn2(x)
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn3(x)
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn4(x)
        #x = F.relu(self.fc4(x))
        x = self.fc4(x)
        x = self.relu4(x)
        return x
