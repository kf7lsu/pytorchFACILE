import os

DEFAULT_SPLIT = (0.6, 0.3, 0.1)

BATCH_SIZE = 500
N_EPOCHS = 5
LEARNING_RATE = 0.01

DATA_FOLDER_PATH = "data"
MODELS_FOLDER_PATH = "models"
BEST_LOSS_PATH = os.path.join(MODELS_FOLDER_PATH, "min_ave_val_loss_quant.txt")

WEIGHT_BW = 6
ACT_BW = 6

PREPROC_MINS =[
                31.0,
                1.0,
                0.0,
                -29.0,
                1.0,
                0.000076189,
                17.317,
                33.448,
                17.927,
                55.531,
                55.538,
                36.765,
                36.762,
                36.761
                ]
PREPROC_MAXES = [
                113.0,
                7.0,
                495.129,
                29.0,
                72.0,
                0.012621,
                65325.984375,
                59779.774,
                143953.485,
                615225.3125,
                254884.40625,
                78382.790,
                39076.9961,
                30473.15821
                ]
POSTPROC_MIN = [0.0]
POSTPROC_MAX = [507.4008]
PREPROC_INCRS_4b = [
                    5.4667,
                    0.40,
                    33.009,
                    3.8667,
                    4.7333,
                    0.00083629,
                    4353.9,
                    3983.1,
                    9595.7,
                    41011.0,
                    16989.0,
                    5223.1,
                    2602.7,
                    2029.1
                    ]
POSTPROC_INCR_4b = [33.8267]