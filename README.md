# pytorchFACILE

## Setup

Using a virtual environment:

    python -m venv ./venv
    source venv/bin/activate
    pip install -r requirements.txt

Using conda:

    conda create --name pytorchFACILE-env python=3.8.5
    conda activate pytorchFACILE-env
    conda install torch
    pip install -r requirements.txt

## Scripts

### utils.py
  
Contains all of the data loading code. Can load a pandas Dataframe, numpy arrays, or torch TensorDatasets.

    import utils
    X, Y = utils.open_p2_df() # open pandas dataframes
    X, Y = utils.load_np_data() # load all data as two numpy arrays
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_split_np_data() # load split data (will used saved split data if found)
    train_set, val_set, test_set, _ = utils.load_split_np_data() # TensorDatasets

If you want to change the train/val/test split, you will need to ```rm data/*.npy```, so that it doesn't use the saved split data with the old split
and splits the data using the new split.

### constants.py

Set default values for functions, so you don't need to fill out the keyword arguments every time you call them. Example:

    import utils
    X, Y = utils.load_np_data() # DATA_FOLDER_PATH="data" from constants.py, so loads data/X_allData.pkl and data/Y_allData.pkl
    X, Y = utils.load_np_data(data_path="other_data") # load other_data/X_allData.pkl and other_data/Y_allData.pkl

Import as ```from constants.py import *```, so you don't need to include ```constants.``` before the variable name. Only create constant variables
in constants.py such that all variables with all uppercase names can be found in constants.py. Otherwise, the import trick will make it frusturating
to find variables.

### train.py

Call ```train(ModelClass)``` to train a model. Models are saved in MODELS_FOLDER_PATH by default if their average validation loss is less than the 
minimum average validation loss found so far. The minimum average validation loss is also saved to a file BEST_LOSS_PATH, which is models/min_ave_val_loss.txt
by default.

### model.py and quant_model.py

Both define pytorch models. In the constructor, each layer is initialized and in forward propagation they are applied to the input x.

### rm_old_models.sh

Call ```./rm_old_models.sh``` to remove old models from models folder, leaving you only with the best model and the min_ave_val_loss.txt files. To only remove
the n oldest models call ```./rm_old_models.sh models $n```. To remove from another folder, call ```./rm_old_models.sh $folder```.
