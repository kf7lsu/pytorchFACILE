import pickle
import os
import numpy as np

DEFAULT_SPLIT = (0.9, 0.05, 0.05)

def _open_p2_data(path):
    """
    Open pickled dataframes from Python2
    """
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()

def load_data(data_path="data"):
    """
    Load data from default paths
    """
    res = []
    for i in ("X", "Y"):
        path = os.path.join(data_path, f"{i}_allData.pkl")
        res.append(_open_p2_data(path).to_numpy())
        print(f"Shape of {i}_total: {res[-1].shape}")
    return res

def split_data(X, Y, split=DEFAULT_SPLIT, save_path=""):
    """
    Split X, Y using the split and save if save_path given.
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

def load_split_data(split=DEFAULT_SPLIT, data_path="data"):
    """
    Load X_train, X_val, X_test, Y_train, Y_val, Y_test if already saved,
    split data again otherwise.
    """
    res = []
    for a in ("X", "Y"):
        for b in ("train", "val", "test"):
            path = os.path.join(data_path, f"{a}_{b}.npy")
            if not os.path.exists(path):
                print(f"Missing {path} file, recalculating split")
                X, Y = load_data(data_path=data_path)
                return split_data(X, Y, split=split, save_path=data_path)
            res.append(np.load(path))

    print("Using saved split data")
    return res
