import os

DEFAULT_SPLIT = (0.6, 0.3, 0.1)

BATCH_SIZE = 500
N_EPOCHS = 100
LEARNING_RATE = 0.03

DATA_FOLDER_PATH = "data"
MODELS_FOLDER_PATH = "models"
BEST_LOSS_PATH = os.path.join(MODELS_FOLDER_PATH, "min_ave_val_loss_quant.txt")

WEIGHT_BW = 4
ACT_BW = 4