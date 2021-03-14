### WAND Debug ###

import wandb
import logConfig, load_data, accuracy_loss, train, preprocessing, plot

train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(N, w, h), n_labels = train_X.shape, len(labels)

# Number of datapoints to train
n = 1000

# Dimension of datapoints
d = w * h

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(d, n_labels, train_X, train_Y, test_X, test_Y)

def mainDebug(config = None):
  run = wandb.init(config=config)
  config = wandb.config

  hl = [config.hidden_layer_size] * config.hidden_layers               # Hidden layers
  ol = [len(train_y[0])]                                               # Output layers
  n_hl = len(hl)

  logConfig.logConfig(config)

  # loss_functions = [ "sq_loss", "cross_entropy"]
  # for loss_function in loss_functions:
  #   W, b = train.train(train_x[:100], train_y[:100], val_x[:100], val_y[:100], d, hl, ol, config, loss_function)
  #   test_acc, test_loss, y, _y = accuracy_loss.get_accuracy_loss_and_prediction(W, b, test_x[:n], test_y[:n], n_hl, config.ac, loss_function)
  #   if loss_function == "cross_entropy":
  #     confusion_matrix= plot.confusion_matrix(labels, y, _y)
  #     wandb.log( { "test_accuracy": test_acc, "test_loss": test_loss ,  "Confusion Matrix": confusion_matrix  } )
  #   else:
  #     wandb.log( { "test_accuracy (Squared Loss)": test_acc, "test_loss (Squared Loss)": test_loss } )

  W, b = train.train(train_x[:100], train_y[:100], val_x[:100], val_y[:100], d, hl, ol, config, "cross_entropy")
  test_acc1, test_loss1, y, _y = accuracy_loss.get_accuracy_loss_and_prediction(W, b, test_x[:n], test_y[:n], n_hl, config.ac, "cross_entropy")

  W, b = train.train(train_x[:100], train_y[:100], val_x[:100], val_y[:100], d, hl, ol, config, "sq_loss")
  test_acc2, test_loss2, y, _y = accuracy_loss.get_accuracy_loss_and_prediction(W, b, test_x[:n], test_y[:n], n_hl, config.ac, "sq_loss")
  wandb.log( { "test_accuracy": test_acc1, "test_loss": test_loss1 , "test_accuracy (Squared Loss)": test_acc2, "test_loss (Squared Loss)": test_loss2  } )

  confusion_matrix= plot.confusion_matrix(labels, y, _y)
  wandb.log( {"Confusion Matrix": confusion_matrix  } )


  run.finish()

sweep_config = {
  "name": "Test Sweep",
  "method": "bayes",
  'metric': {
      'name': 'accuracy',
      'goal': 'maximize'
  },
  "early_terminate" : {
      "type": "hyperband",
      "min_iter": 3
  },
  "parameters": {
        "learning_rate": {
            "values": [0.001]
        },
        "epochs" :{
            "values" : [10]
        },
        "hidden_layers": {
            "values": [5]
        },
        "hidden_layer_size": {
            "values": [128]
        },
        "optimiser": {
            "values": ["sgd", "mgd", "nag", "rmsprop", "adam", "nadam"]
        },
        "batch_size": {
            "values": [64]
        },
        "init_strategy": {
            "values": ["xavier"]
        },
        "ac": {
            "values": ["tanh"]
        },
         "weight_decay": {
            "values": [0]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Test 1")
wandb.agent(sweep_id, mainDebug, count = 1)