import os

DEFAULT_SPLIT = (0.6, 0.3, 0.1)

BATCH_SIZE = 1
N_EPOCHS = 20
LEARNING_RATE = 0.001

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
                    1.3015873015873016,
                    0.09523809523809523, 
                    7.859190476190476, 
                    0.9206349206349206, 
                    1.126984126984127, 
                    0.0001991239841269841, 
                    1036.645513888889, 
                    948.354380952381, 
                    2284.6913968253966, 
                    9764.599706349207, 
                    4044.9026706349205, 
                    1243.5876984126983, 
                    619.6862555555555, 
                    483.1174160317461
                   ]
POSTPROC_INCR_4b = [8.053980952380952]