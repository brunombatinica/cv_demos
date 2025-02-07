import os
import tempfile
import time

import mlflow

import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

# def train_function(config):
#     width, height = config["width"], config["height"]

#     for step in range(config.get("steps", 100)):
#         # Iterative training function - can be any arbitrary training procedure
#         intermediate_score = evaluation_fn(step, width, height)
#         # Feed the score back to Tune.
#         train.report({"iterations": step, "mean_loss": intermediate_score})
#         time.sleep(0.1)

# def tune_with_callback(mlflow_tracking_uri, finish_fast=False):
#     tuner = tune.Tuner(
#         train_function,
#         tune_config=tune.TuneConfig(num_samples=2),
#         run_config=train.RunConfig(
#             name="mlflow",
#             callbacks=[
#                 MLflowLoggerCallback(
#                     tracking_uri=mlflow_tracking_uri,
#                     experiment_name="mlflow_callback_example",
#                     save_artifact=True,
#                 )
#             ],
#         ),
#         param_space={
#             "width": tune.randint(10, 100),
#             "height": tune.randint(0, 100),
#             "steps": 5 if finish_fast else 100,
#         },
#     )
#     results = tuner.fit()

def train_function_mlflow(config):
    tracking_uri = config.pop("tracking_uri", None)
    setup_mlflow(
        config,
        experiment_name="setup_mlflow_example",
        tracking_uri=tracking_uri,
    )

    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Log the metrics to mlflow
        mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
        # Feed the score back to Tune.
        train.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(0.1)

def tune_with_setup(mlflow_tracking_uri, finish_fast=False):
    # Set the experiment, or create a new one if does not exist yet.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name="setup_mlflow_example")

    tuner = tune.Tuner(
        train_function_mlflow,
        tune_config=tune.TuneConfig(num_samples=2),
        run_config=train.RunConfig(
            name="mlflow",
        ),
        param_space={
            "width": tune.randint(10, 100),
            "height": tune.randint(0, 100),
            "steps": 5 if finish_fast else 100,
            "tracking_uri": mlflow.get_tracking_uri(),
        },
    )
    results = tuner.fit()

# ray.init()
tune_with_setup("file:///C:/Users/bruno/OneDrive/Documents/Code/projects/cv_demos/mlruns",finish_fast=True)
#tune_with_callback("file:///C:/Users/bruno/OneDrive/Documents/Code/projects/cv_demos/mlruns",finish_fast=True)