import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    # ModelExp4 in https://github.com/JackDinsmore/FACILE/blob/master/train-models.py

    def __init__(self, n_features):
        super(BaseNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_features, momentum=0.6)
        self.fc1 = nn.Linear(n_features, 31)
        self.bn2 = nn.BatchNorm1d(31, momentum=0.6)
        self.fc2 = nn.Linear(31, 11)
        self.bn3 = nn.BatchNorm1d(11, momentum=0.6)
        self.fc3 = nn.Linear(11, 3)
        self.fc4 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)

if __name__=="__main__":
    print(BaseNet(14))
