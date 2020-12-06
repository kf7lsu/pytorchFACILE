import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import utils
from model import FACILENet as Net
from constants import *

def train(batch_size=BATCH_SIZE, n_epochs=N_EPOCHS):
    train_set, val_set, test_set, n_features = utils.load_torch_datasets()

    gen_params = {
                "batch_size": batch_size,
                "shuffle": True,
            }

    train_gen = DataLoader(train_set, **gen_params)
    val_gen = DataLoader(val_set, **gen_params)

    model = Net(n_features=n_features)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in trainable_params])
    print(f"# of Model parameters: {n_params}")

    for epoch in range(n_epochs):
        continue
        print(f"\n{'='*30}\n")

if __name__ == "__main__":
    train()
