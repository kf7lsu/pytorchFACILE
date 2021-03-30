import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime
import sys
import numpy as np
import pickle

import utils
from metrics import Metrics
from model import BaseNet
from quant_model import QuantNet
from constants import *

from processing_for_train import FACILE_preproc as preproc
from processing_for_train import FACILE_postproc as postproc
from processing_for_train import FACILE_preproc_out as preproc_out

def train(model_class, metrics=None, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, 
        models_folder_path=MODELS_FOLDER_PATH, quantized=False):
    #if quantized:
    #    train_set, val_set, test_set, n_features = utils.load_torch_datasets_quant()
    #else:
    train_set, val_set, test_set, n_features = utils.load_torch_datasets()

    gen_params = {
                "batch_size": BATCH_SIZE,
                "shuffle": True,
            }

    train_gen = DataLoader(train_set, **gen_params)
    val_gen = DataLoader(val_set, **gen_params)
    print(f"Number of batches per epoch: {len(train_gen)}")

    model = model_class(n_features=n_features).float()
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in trainable_params])
    print(f"# of Model parameters: {n_params}")
    print(f"# of features: {n_features}")

    min_ave_val_loss = utils.load_num(BEST_LOSS_PATH)
    best_model = None
    
    if quantized:
        #ensure brevitas model is in training mode
        model.train()
    
    print(f"\n{'='*30}\n")
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")

        total_train_loss = 0
        total_val_loss = 0
        total_train_loss_quant = 0
        total_val_loss_quant = 0
        n_train_samples = 0
        n_val_samples = 0

        for train_batch, labels_batch in train_gen:
            n_train_samples += train_batch.shape[0]

            if quantized:
                labels_batch_q = labels_batch
                labels_batch = preproc_out(labels_batch.float())
                output_batch = preproc(train_batch.float())
                output_batch = model(output_batch)
                #output_batch = postproc(output_batch).float()
                #output_batch = torch.trunc(output_batch)
                output_batch_quant = torch.round(output_batch)
                output_batch_quant = postproc(output_batch_quant)
                q_loss = loss_fn(output_batch_quant.float(), labels_batch_q.float())
                total_train_loss_quant += q_loss.item()
            else:
                output_batch = model(train_batch.float())
            loss = loss_fn(output_batch.float(), labels_batch.float())
            total_train_loss += loss.item()

            optimizer.zero_grad() # clear previous gradients
            loss.backward() # compute gradients

            optimizer.step() # update weights using computed gradients

        for val_batch, labels_batch in val_gen:
            n_val_samples += val_batch.shape[0]

            #output_batch = model(val_batch.float())
            if quantized:
                labels_batch_q = labels_batch
                labels_batch = preproc_out(labels_batch.float())
                output_batch = preproc(val_batch.float())
                output_batch = model(output_batch)
                #output_batch = postproc(output_batch).float()
                #output_batch = torch.trunc(output_batch)
                output_batch_quant = torch.round(output_batch)
                output_batch_quant = postproc(output_batch_quant)
                q_loss = loss_fn(output_batch_quant.float(), labels_batch_q.float())
                total_val_loss_quant += q_loss.item()
            else:
                output_batch = model(val_batch.float())
            total_val_loss += loss_fn(output_batch.float(), 
                    labels_batch.float()).item()

        ave_train_loss = total_train_loss / n_train_samples
        ave_val_loss = total_val_loss / n_val_samples
        ave_train_loss_q = total_train_loss_quant / n_train_samples
        ave_val_loss_q = total_val_loss_quant / n_val_samples

        if (metrics):
            metrics.train_losses.append(ave_train_loss)
            metrics.val_losses.append(ave_val_loss)
            metrics.train_losses_quant.append(ave_train_loss_q)
            metrics.val_losses_quant.append(ave_val_loss_q)

        print(f"Ave Train Loss: {ave_train_loss}")
        print(f"Ave Val Loss: {ave_val_loss}")
        print(f"Ave Q Train Loss: {ave_train_loss_q}")
        print(f"Ave Q Val Loss: {ave_val_loss_q}")

        saved_model = False
        if ave_val_loss < min_ave_val_loss or min_ave_val_loss == -1:
            saved_model = True
            # Save best models
            name = datetime.datetime.now().strftime("%b-%d-%I%M%p-%G-%f")
            filepath = os.path.join(models_folder_path, f"{name}.pkl")
            best_model = model
            print(type(best_model))
            if not quantized:  #brevitas doesn't like pickle
                with open(filepath, "wb+") as f:
                    pickle.dump(model, f)

            # Save ave loss as well for future runs
            utils.save_num(ave_val_loss, BEST_LOSS_PATH)

            min_ave_val_loss = ave_val_loss

        print(f"Min Ave Val Loss: {min_ave_val_loss}")
        if saved_model:
            print("Saved model")

        print(f"\n{'='*30}\n")
    if quantized:  #just spit out the best model, can finn export later
        print(type(best_model))
        return best_model

def main():
    # Hide stack trace when keyboard interrupt
    metrics = Metrics()
    try:
        train(BaseNet, metrics=metrics)
        # train(QuantNet, metrics=metrics, models_folder="quant_models")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        metrics.plot_losses()

if __name__ == "__main__":
    main()
