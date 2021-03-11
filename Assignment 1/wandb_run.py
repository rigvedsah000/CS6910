### WAND Run ###

import wandb
import logConfig, load_data, accuracy_loss, train, preprocessing

train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(N, w, h), n_labels = train_X.shape, len(labels)

# Dimension of datapoints
d = w * h

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(d, n_labels, train_X, train_Y, test_X, test_Y)


def main(config = None):
  run = wandb.init(config=config, resume=True)
  config = wandb.config

  hl = [config.hidden_layer_size] * config.hidden_layers               # Hidden layers
  ol = [len(train_y[0])]                                               # Output layers
  n_hl = len(hl)

  logConfig.logConfig(config)

  W, b = train.train(train_x, train_y, val_x, val_y, d, hl, ol, config)

  val_acc, val_loss = accuracy_loss.get_accuracy_and_loss(W, b, val_x, val_y, n_hl, config.ac)

  wandb.log( { "val_accuracy": val_acc } )

  run.finish()

sweep_config = {
  "name": "Sweep 1",
  "method": "bayes",
  'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
  },
  "early_terminate" : {
      "type": "hyperband",
      "min_iter": 3,
      "eta": 3,
  },
  "parameters": {
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        "epochs" :{
            "values": [5, 10]
        },
        "hidden_layers": {
            "values": [3, 4, 5]
        },
        "hidden_layer_size": {
            "values": [64, 128]
        },
        "optimiser": {
            "values": ["sgd", "mgd", "nag"]
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "init_strategy": {
            "values": ["random", "xavier"]
        },
        "ac": {
            "values": ["sig", "tanh", "relu"]
        },
         "weight_decay": {
            "values": [0, 0.0005, 0.5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment 1")
wandb.agent(sweep_id, project="Assignment 1", function=main)