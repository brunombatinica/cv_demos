import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from ray import tune
from ray.air.integrations.mlflow import setup_mlflow
import mlflow
from datetime import datetime
import sys
import os
sys.path.append( os.path.join(os.path.abspath(".").split("src")[0], "src"))

#print(sys.path)
from cifar10_tune.data.cifar10_dataloaders import get_cifar10_dataloaders
from cifar10_tune.models.cnn import CNN

from pdb import set_trace as bp

def train_one_epoch(model=None, 
                    optimizer=None, 
                    loss_fn=None, 
                    train_dataloader=None, 
                    val_dataloader=None,
                    progress=False,
                    verbose=False,
                    device=None,
                    epoch_n=None) -> tuple:
    '''
    1 epoch cycle of training for a general nn model
    args:
        model: nn.Module, model
        optimizer: optim.Optimizer, optimizer
        loss_fn: nn.Module, loss function
        train_dataloader: DataLoader, train dataloader
        val_dataloader: DataLoader, validation dataloader
        progress: bool, whether to show progress bar
        verbose: bool, whether to print verbose output
        device: torch.device, device to run the model on
        epoch_n: int, epoch number
    returns:
        train_loss: list, train loss
        val_loss: list, validation loss
    '''
    model.train()
    train_loss = []
    train_dataloader_ = tqdm(train_dataloader,leave=False) if progress else train_dataloader
    for batch_x, batch_y in train_dataloader_:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device) # move intputs to GPU

        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    if progress: train_dataloader_.close()
    
    # validation loop
    model.eval()
    with torch.no_grad():
        val_labels = []
        predictions = []
        val_loss = []
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # move intputs to GPU
            
            output = model(batch_x)
            loss = loss_fn(output, batch_y)

            predictions.append(output.argmax(dim=1).detach().cpu().numpy())
            val_labels.append(batch_y.detach().cpu().numpy())
            val_loss.append(loss.item())
        predictions = np.concatenate(predictions)
        val_labels = np.concatenate(val_labels)

    # log metrics to mlflow
    try:
        time = tune.get_reported_time()
    except:
        time = 0
    train_loss_ = np.mean(train_loss)
    val_loss_ = np.mean(val_loss)
    val_acc_ = np.mean(predictions == val_labels)

    # log metrics to mlflow
    mlflow.log_metrics({"train_loss": train_loss_,
                         "val_loss": val_loss_, 
                         "val_acc": val_acc_,
                         "compute_time": time
                         },
                         step=epoch_n)

    # report metrics to Ray Tune
    #print(f"val_loss: {np.mean(val_loss)}, type: {type(np.mean(val_loss))}")
    tune.report({"train_loss": train_loss_,
                "val_loss": val_loss_, 
                "val_acc": val_acc_,
                "compute_time": time})
    if verbose: print(f"train_loss: {train_loss_}, val_loss: {val_loss_}, val_acc: {val_acc_}, compute_time: {time}")
    return train_loss, val_loss
        


def train_model(config,
          progress=False,
          verbose=False,
          seed=None,
          single_run=False):
    '''
    train the model
    args:
        config: dict, config
        progress: bool, whether to show progress bar
        verbose: bool, whether to print verbose output
        seed: int, seed for reproducibility
        single_run: bool, whether this is being run as a single run or part of a hyperparameter search
    returns:
        train_losses: list, train losses
        val_losses: list, val losses
    '''
    #bp()

    # initialize mlflow
    run_name =f"CNN_{config['optimizer']}_{config['conv_layers']}layers_{config['hidden_dim']}dim_{config['dropout']}drop_{config['learning_rate']}lr_{datetime.now().strftime('%m_%d__%H_%M')}"
    if verbose: print(run_name)
    
    # mlflow
    mlflow.start_run()
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_tag("mlflow.runName", run_name)  # Enforce run name in MLflow UI
    mlflow.log_params(config)
    
    if verbose:print("MLflow is using:", mlflow.get_tracking_uri())
    # set seed for reproducibility
    if seed: 
        torch.manual_seed(seed) # set seed for reproducibility
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # load dataloaders
    train_dataloader, val_dataloader, _ = get_cifar10_dataloaders(config["batch_size"])
    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    model = CNN(conv_layers=config["conv_layers"], 
                hidden_dim=config["hidden_dim"], 
                dropout=config["dropout"]).to(device)
    print(f"model device: {next(model.parameters()).device}")
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config["optimizer"].capitalize())(model.parameters(), lr=config["learning_rate"])
    # train loop
    train_losses = []
    val_losses = []
    epoch_range = tqdm(range(config["epochs"])) if progress else range(config["epochs"])
    for epoch in epoch_range:
        train_loss, val_loss = train_one_epoch(model = model,
                                                train_dataloader = train_dataloader,
                                                val_dataloader = val_dataloader,
                                                loss_fn = criterion,
                                                optimizer = optimizer,
                                                device = device,
                                                progress = False,
                                                verbose = verbose,
                                                epoch_n = epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if progress:
            tqdm.write(f"Epoch {epoch} train loss: {np.mean(train_loss)} val loss: {np.mean(val_loss)}")
        elif not progress and verbose:
            print(f"Epoch {epoch} train loss: {np.mean(train_loss)} val loss: {np.mean(val_loss)}")
        else:
            pass
    
    # if single_run:
    #     mlflow.end_run()
    #return {"train_losses": train_losses, "val_losses": val_losses}


if __name__ == "__main__":
    # for debugging
    config = {
        "batch_size": 32,
        "conv_layers": 2,
        "hidden_dim": 64,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 3,
        "optimizer": "adam"
    }
    train_model(config,progress=True,verbose=True,seed=42,single_run=True)
    #print(f"train loss: {np.array(train_results['train_losses']).mean(axis=1).min()} val loss: {np.array(train_results['val_losses']).mean(axis=1).min()}")














