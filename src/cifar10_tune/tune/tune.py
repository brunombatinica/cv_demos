import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from ray.tune.integration.mlflow import mlflow_mixin
import mlflow

import sys
import os
SRC_PATH = os.path.join(os.path.abspath(".").split("src")[0], "src")
sys.path.append( SRC_PATH )


from cifar10_tune.train.train import train_model

MLFLOW_TRACKING_URI = "file:///C:/Users/bruno/OneDrive/Documents/Code/projects/cv_demos/mlruns"
env_vars =  {"PYTHONPATH": SRC_PATH,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": "CIFAR10"}
os.environ.update(env_vars)
ray.init(runtime_env={"env_vars": env_vars},
         num_gpus=1)

# Set the experiment, or create a new one if does not exist yet.
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

# search space for hyperparams
search_space = {
    "batch_size": tune.choice([16, 32, 64]),
    "dropout": tune.choice([0.0, 0.1, 0.2, 0.5]), # can replace with tune.uniform(0.0, 0.5)
    "conv_layers": tune.choice([1, 2, 3, 4]),
    "hidden_dim": tune.choice([32, 64, 128, 256]),
    "epochs": 10,
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "optimizer": "adam"

}

asha_scheduler = ASHAScheduler(
    max_t=20,
    grace_period=3,
    reduction_factor=2
)

trainable = tune.with_resources(train_model, resources={"cpu": 10, "gpu": 1})
tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    run_config=ray.air.RunConfig(
        name="CIFAR10",
    ),
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=20,
        scheduler=asha_scheduler,
    ),
)

results = tuner.fit()

best_result = results.get_best_result("val_loss", "min")
print("Best trial config: {}".format(best_result.config))
print("Best trial final validation loss: {}".format(
    best_result.metrics["val_loss"]))
print("Best trial final validation accuracy: {}".format(
    best_result.metrics["val_acc"]))

## could use a test set here
