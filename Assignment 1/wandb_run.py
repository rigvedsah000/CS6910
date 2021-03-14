### WAND Run ###

import wandb
import logConfig, load_data, accuracy_loss, train, preprocessing, plot

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

  name = "hl_" + str(config.hidden_layers) + "_bs_" + str(config.batch_size) + "_ac_" + config.ac
  run.name = name

  logConfig.logConfig(config)

  # Set Loss function here
  loss_functions = [ "cross_entropy", "sq_loss" ]
  
  W, b = train.train(train_x, train_y, val_x, val_y, d, hl, ol, config, loss_functions[0])
  test_acc, test_loss, y, _y = accuracy_loss.get_accuracy_loss_and_prediction(W, b, test_x, test_y, n_hl, config.ac,loss_functions[0])
  confusion_matrix= plot.confusion_matrix(labels, y, _y)
  
  if len(loss_functions) == 2:
    W_, b_ = train.train(train_x, train_y, val_x, val_y, d, hl, ol, config, loss_functions[1])
    test_acc_, test_loss_ = accuracy_loss.get_accuracy_and_loss(W_, b_, test_x, test_y, n_hl, config.ac, loss_functions[1])
    wandb.log( { "test_accuracy": test_acc, "test_loss": test_loss , "test_accuracy (Squared Loss)": test_acc_, "test_loss (Squared Loss)": test_loss_ , "Confusion Matrix": confusion_matrix  } )

  else:
    wandb.log( { "test_accuracy": test_acc, "test_loss": test_loss }, {"Confusion Matrix": confusion_matrix  } )

  run.finish()

sweep_config = {
  "name": "Final Model 3",
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
            "values": [0.001]
        },
        "epochs" :{
            "values": [5, 10]
        },
        "hidden_layers": {
            "values": [4, 5]
        },
        "hidden_layer_size": {
            "values": [64, 128]
        },
        "optimiser": {
            "values": ["adam", "rmsprop", "nadam"]
        },
        "batch_size": {
            "values": [64]
        },
        "init_strategy": {
            "values": ["xavier"]
        },
        "ac": {
            "values": ["tanh", "relu", "sig"]
        },
         "weight_decay": {
            "values": [0]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment 1")
wandb.agent(sweep_id, project="Assignment 1", function=main)
