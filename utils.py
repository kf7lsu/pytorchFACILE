import pickle
import numpy as np

def _open_p2_data(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def load_data(X_path="data/X_alldata.pkl", Y_path="data/Y_alldata.pkl"):
    """
    Load data from default paths
    """
    return _open_p2_data(X_path), _open_p2_data(Y_path)

def split_data(X, Y, split=(0.8, 0.1, 0.1), save_path=""):
    """
    Split X, Y using the split and save if save_path given.
    """
    total = np.concatenate((X, Y), axis=1)
    np.random.shuffle(total)
    input_n = (int) ((1 - split[0]) * total.shape[0])
    val_n = ((1 - split[0] - split[1]) * total.shape[0])

print(load_data())
