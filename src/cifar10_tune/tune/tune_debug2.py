import os
import tempfile
import time
import numpy as np
import torch
import mlflow


import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

from ray.tune.schedulers import ASHAScheduler
# from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

import sys
import os
SRC_PATH = os.path.join(os.path.abspath(".").split("src")[0], "src")
sys.path.append( SRC_PATH )


from cifar10_tune.data.cifar10_dataloaders import get_cifar10_dataloaders
from cifar10_tune.models.cnn import CNN
from cifar10_tune.train.train import train_model


# def train_one_epoch(model=None, 
#                     optimizer=None, 
#                     loss_fn=None, 
#                     train_dataloader=None, 
#                     val_dataloader=None,
#                     progress=False,
#                     verbose=False,
#                     device=None,
#                     epoch_n=None) -> tuple:
#     '''
#     1 epoch cycle of training for a general nn model
#     args:
#         model: nn.Module, model
#         optimizer: optim.Optimizer, optimizer
#         loss_fn: nn.Module, loss function
#         train_dataloader: DataLoader, train dataloader
#         val_dataloader: DataLoader, validation dataloader
#         progress: bool, whether to show progress bar
#         verbose: bool, whether to print verbose output
#         device: torch.device, device to run the model on
#         epoch_n: int, epoch number
#     returns:
#         train_loss: list, train loss
#         val_loss: list, validation loss
#     '''
#     model.train()
#     train_loss = []
#     for batch_x, batch_y in train_dataloader:

#         batch_x, batch_y = batch_x.to(device), batch_y.to(device) # move intputs to GPU

#         optimizer.zero_grad()
#         output = model(batch_x)
#         loss = loss_fn(output, batch_y)
#         loss.backward()
#         optimizer.step()

#         train_loss.append(loss.item())
    
#     # validation loop
#     model.eval()
#     with torch.no_grad():
#         val_labels = []
#         predictions = []
#         val_loss = []
#         for batch_x, batch_y in val_dataloader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device) # move intputs to GPU
            
#             output = model(batch_x)
#             loss = loss_fn(output, batch_y)

#             predictions.append(output.argmax(dim=1).detach().cpu().numpy())
#             val_labels.append(batch_y.detach().cpu().numpy())
#             val_loss.append(loss.item())
#         predictions = np.concatenate(predictions)
#         val_labels = np.concatenate(val_labels)

#     # log metrics to mlflow
#     try:
#         time = tune.get_reported_time()
#     except:
#         time = 0
#     train_loss_ = np.mean(train_loss)
#     val_loss_ = np.mean(val_loss)
#     val_acc_ = np.mean(predictions == val_labels)

#     # log metrics to mlflow
#     mlflow.log_metrics({"train_loss": train_loss_,
#                          "val_loss": val_loss_, 
#                          "val_acc": val_acc_,
#                          "compute_time": time
#                          },
#                          step=epoch_n)

#     # report metrics to Ray Tune
#     #print(f"val_loss: {np.mean(val_loss)}, type: {type(np.mean(val_loss))}")
#     tune.report({"train_loss": train_loss_,
#                 "val_loss": val_loss_, 
#                 "val_acc": val_acc_,
#                 "compute_time": time})
#     if verbose: print(f"train_loss: {train_loss_}, val_loss: {val_loss_}, val_acc: {val_acc_}, compute_time: {time}")
#     return train_loss, val_loss
        


# def train_model(config,
#           progress=False,
#           verbose=False,
#           seed=None,
#           single_run=False):
#     '''
#     train the model
#     args:
#         config: dict, config
#         progress: bool, whether to show progress bar
#         verbose: bool, whether to print verbose output
#         seed: int, seed for reproducibility
#         single_run: bool, whether this is being run as a single run or part of a hyperparameter search
#     returns:
#         train_losses: list, train losses
#         val_losses: list, val losses
#     '''
#     #bp()

#     # initialize mlflow
#     run_name =f"CNN_{config['optimizer']}_{config['conv_layers']}layers_{config['hidden_dim']}dim_{config['dropout']}drop_{config['learning_rate']}lr_{datetime.now().strftime('%m_%d__%H_%M')}"
#     if verbose: print(run_name)
#     # if not single_run:
#     #     setup_mlflow(
#     #         config,
#     #         experiment_name="CIFAR10",
#     #         tracking_uri="file:./mlruns",
#     #         run_name=run_name
#     #     )
#     # else:
#     #     mlflow.set_experiment("CIFAR10")
#     #     mlflow.start_run(run_name=run_name)
#     #     mlflow.log_params(config)
#     #     mlflow.set_tag("model", "CNN")
#     setup_mlflow(
#         config,
#         experiment_name="CIFAR10",
#         tracking_uri="file:C:/Users/bruno/OneDrive/Documents/Code/projects/cv_demos/mlruns",
#         run_name=run_name
#     )
#     print("MLflow is using:", mlflow.get_tracking_uri())
#     # set seed for reproducibility
#     if seed: 
#         torch.manual_seed(seed) # set seed for reproducibility
#         np.random.seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     # load dataloaders
#     train_dataloader, val_dataloader, _ = get_cifar10_dataloaders(config["batch_size"])
#     # initialize model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CNN(conv_layers=config["conv_layers"], 
#                 hidden_dim=config["hidden_dim"], 
#                 dropout=config["dropout"]).to(device)
#     # define loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, config["optimizer"].capitalize())(model.parameters(), lr=config["learning_rate"])
    
#     # train loop
#     train_losses = []
#     val_losses = []
#     epoch_range = tqdm(range(config["epochs"])) if progress else range(config["epochs"])
#     for epoch in epoch_range:
#         train_loss, val_loss = train_one_epoch(model = model,
#                                                 train_dataloader = train_dataloader,
#                                                 val_dataloader = val_dataloader,
#                                                 loss_fn = criterion,
#                                                 optimizer = optimizer,
#                                                 device = device,
#                                                 progress = False,
#                                                 verbose = verbose,
#                                                 epoch_n = epoch)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

# def train_model(config):
#     tracking_uri = config.pop("tracking_uri", None)
#     setup_mlflow(
#         config,
#         experiment_name="setup_mlflow_example",
#         tracking_uri=tracking_uri,
#     )

#     # Hyperparameters
#     width, height = config["width"], config["height"]

#     for step in range(config.get("steps", 100)):
#         # Iterative training function - can be any arbitrary training procedure
#         intermediate_score = evaluation_fn(step, width, height)
#         # Log the metrics to mlflow
#         mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
#         # Feed the score back to Tune.
#         train.report({"iterations": step, "mean_loss": intermediate_score})
#         time.sleep(0.1)


