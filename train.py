from torch.utils.data import DataLoader

from model import FACILENet as Net
from constants import *
import utils

def train(batch_size=BATCH_SIZE, n_epochs=N_EPOCHS):
    train_set, val_set, test_set = utils.load_torch_datasets()

    gen_params = {
                "batch_size": batch_size,
                "shuffle": True,
            }

    train_gen = DataLoader(train_set, **gen_params)
    val_gen = DataLoader(val_set, **gen_params)

    for epoch in range(n_epochs):
        print(f"\n{'='*30}\n")


if __name__ == "__main__":
    train()
