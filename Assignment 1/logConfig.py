import wandb

def logConfig(config):
  wandb.log({
      "epochs": config.epochs,
      "ac": config.ac,
      "hidden_layers": config.hidden_layers,
      "hidden_layer_size": config.hidden_layer_size,
      "learning_rate": config.learning_rate,
      "optimiser": config.optimiser,
      "init_strategy": config.init_strategy,
      "batch_size": config.batch_size,
  })