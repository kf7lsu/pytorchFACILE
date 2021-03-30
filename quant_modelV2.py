import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity
from brevitas.core.quant import QuantType
from constants import WEIGHT_BW, ACT_BW

class QuantNet_opt(nn.Module):
    def __init__(self, n_features):
        super(QuantNet_opt, self).__init__()
        #self.bn0 = nn.BatchNorm1d(n_features, affine=False)
        #self.relu0 = QuantReLU(bit_width=ACT_BW)
        #self.qinp = QuantIdentity(quant_type=QuantType.INT, bit_width=ACT_BW)
        self.fc1 = QuantLinear(n_features, 31, bias=True, weight_bit_width=WEIGHT_BW)
        #self.bn1 = nn.BatchNorm1d(31, affine=False)
        self.relu1 = QuantReLU(bit_width=ACT_BW)
        self.fc2 = QuantLinear(31, 11, bias=True, weight_bit_width=WEIGHT_BW)
        #self.bn2 = nn.BatchNorm1d(11, affine=False)
        self.relu2 = QuantReLU(bit_width=ACT_BW)
        self.fc3 = QuantLinear(11, 3, bias=True, weight_bit_width=WEIGHT_BW)
        #self.bn3 = nn.BatchNorm1d(3, affine=False)
        self.relu3 = QuantReLU(bit_width=ACT_BW)
        self.fc4 = QuantLinear(3, 1, bias=True, weight_bit_width=WEIGHT_BW)
        #self.bn4 = nn.BatchNorm1d(1, affine=False)
        self.relu4 = QuantReLU(bit_width=ACT_BW)
        #self.qout = QuantIdentity(quant_type=QuantType.INT, bit_width=ACT_BW, min_val=0.0, max_val=15.0)

    def forward(self, x):
        #x = self.bn0(x)
        #x = self.relu0(x)
        #x = self.qinp(x)
        x = self.fc1(x)
        #x = self.bn1(x)
        #x = F.relu(self.fc1(x))
        x = self.relu1(x)
        #x = self.drop1(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        #x = F.relu(self.fc2(x))
        x = self.relu2(x)
        x = self.fc3(x)
        #x = self.bn3(x)
        #x = F.relu(self.fc3(x))
        x = self.relu3(x)
        x = self.fc4(x)
        #x = self.bn4(x)
        #x = F.relu(self.fc4(x))
        x = self.relu4(x)
        #x = self.qout(x)
        #x = torch.round(x)
        return x
