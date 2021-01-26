import pickle
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
#from processing_for_train import FACILE_preproc, FACILE_preproc_out

from constants import *

def open_p2_df(path):
    """
    Open pickled pandas dataframes from Python2
    """
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()

def load_np_data(data_path=DATA_FOLDER_PATH):
    """
    Load data from default paths
    """
    res = []
    for i in ("X", "Y"):
        path = os.path.join(data_path, f"{i}_allData.pkl")
        if i == "X":
            res.append(open_p2_df(path).to_numpy())

        if i == "Y":
            arr = open_p2_df(path)["genE"].to_numpy()
            arr = np.reshape(arr, (arr.shape[0], 1))
            res.append(arr)

        print(f"Shape of {i}_total: {res[-1].shape}")
    return res

def split_np_data(X, Y, split=DEFAULT_SPLIT, save_path=""):
    """
    Split X, Y np arrays using the split and save if save_path given.
    """
    print("Splitting data...")
    total = np.concatenate((X, Y), axis=1)
    np.random.shuffle(total)

    test_n = (int) (split[2] * total.shape[0])
    val_n = (int) (split[1] * total.shape[0])
    input_n = total.shape[0] - test_n - val_n
    print(f"Split input/val/test samples: {input_n}/{val_n}/{test_n}")

    X_train = total[:input_n, :-Y.shape[1]]
    X_val = total[input_n:input_n + val_n, :-Y.shape[1]]
    X_test = total[input_n + val_n:, :-Y.shape[1]]

    Y_train = total[:input_n, -Y.shape[1]:]
    Y_val = total[input_n:input_n + val_n, -Y.shape[1]:]
    Y_test = total[input_n + val_n:, -Y.shape[1]:]

    for a in ("X", "Y"):
        for b in ("train", "val", "test"):
            name = f"{a}_{b}"
            arr = locals()[name]
            print(f"{name} shape: {arr.shape}")
            if save_path:
                np.save(os.path.join(save_path, name), arr)

    if save_path:
        print("Saved split data")

    print("Done splitting data")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def load_split_np_data(split=DEFAULT_SPLIT, data_path=DATA_FOLDER_PATH):
    """
    Load X_train, X_val, X_test, Y_train, Y_val, Y_test numpy arrays
    if already saved, split data again otherwise.
    """
    res = []
    for a in ("X", "Y"):
        for b in ("train", "val", "test"):
            name = f"{a}_{b}"
            path = os.path.join(data_path, f"{name}.npy")
            if not os.path.exists(path):
                print(f"Missing {path} file, recalculating split")
                X, Y = load_np_data(data_path=data_path)
                return split_np_data(X, Y, split=split, save_path=data_path)
            arr = np.load(path)
            print(f"{name} shape: {arr.shape}")
            res.append(arr)

    print("Using saved split data")
    return res

def load_torch_datasets(split=DEFAULT_SPLIT, data_path=DATA_FOLDER_PATH):
    """
    Load train_set, val_set, test_set as torch datasets from saved np data
    and return number of features.
    """
    X_train, X_val, X_test, Y_train, Y_val, Y_test = map(torch.from_numpy, load_split_np_data(split=split, data_path=data_path))

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)
    test_set = TensorDataset(X_test, Y_test)

    return train_set, val_set, test_set, X_train.shape[1]

#def load_torch_datasets_quant(split=DEFAULT_SPLIT, data_path=DATA_FOLDER_PATH):
#    """
#    Load train_set, val_set, test_set as torch datasets from saved np data
#    and return number of features.
#    """
#    X_train, X_val, X_test, Y_train, Y_val, Y_test = map(torch.from_numpy, load_split_np_data(split=split, data_path=data_path))

    #preprocess the training data so that all values range between 0 and 1
#    X_train = FACILE_preproc(X_train)
#    X_test = FACILE_preproc(X_test)
#    X_val = FACILE_preproc(X_val)
#    Y_train = FACILE_preproc_out(Y_train)
#    Y_test = FACILE_preproc_out(Y_test)
#    Y_val = FACILE_preproc_out(Y_val)
    
#    train_set = TensorDataset(X_train, Y_train)
#    val_set = TensorDataset(X_val, Y_val)
#    test_set = TensorDataset(X_test, Y_test)

#    return train_set, val_set, test_set, X_train.shape[1]

def save_num(n, filepath):
    """ Save number to file """
    with open(filepath, "w+") as f:
        f.write(str(n))

def load_num(filepath):
    """ Read number from file, returning -1 if not found """
    if not os.path.isfile(filepath):
        return -1

    with open(filepath, "r") as f:
        return float(f.read())
