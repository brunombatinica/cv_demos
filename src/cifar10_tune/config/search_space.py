from ray import tune

# search space for hyperparams
search_space = {
    "batch_size": tune.choice([16, 32, 64]),
    "dropout": tune.choice(0.0, 0.2, 0.5), # can replace with tune.uniform(0.0, 0.5)
    "conv_layers": tune.choice([1, 2, 3]),
    "hidden_dim": tune.choice([32, 64]),
    "epochs": 10,
    "learning_rate": 1e-1, #tune.loguniform(1e-4, 1e-1),
    "optimizer": "adam"
}




