### WAND Debug ###

import wandb
import logConfig, load_data, accuracy_loss, train, preprocessing, plot

train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(N, w, h), n_labels = train_X.shape, len(labels)

# Number of datapoints to train
n = 10000

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

  W, b = train.train(train_x[:54000], train_y[:54000], val_x[:6000], val_y[:6000], d, hl, ol, config)

  test_acc, test_loss, y, _y = accuracy_loss.get_accuracy_loss_and_prediction(W, b, test_x[:n], test_y[:n], n_hl, config.ac)

  confusion_matrix = plot.confusion_matrix(labels, y, _y)
  
  wandb.log( { "test_accuracy": test_acc, "test_loss": test_loss, "Confusion Matrix": confusion_matrix } )

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
            "values": ["nadam"]
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

sweep_id = wandb.sweep(sweep_config, project="Test 2")
wandb.agent(sweep_id, mainDebug, count = 1)