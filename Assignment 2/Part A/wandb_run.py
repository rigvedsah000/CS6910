### WAND Run ###
import numpy as np
import wandb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

import load_data, preprocessing

# Load Data
train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(h, w, d), n_labels = train_X[0].shape, len(labels)

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(h, w, d, n_labels, train_X, train_Y, test_X, test_Y)

# Pre-work for Augmentation
datagen = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2
)

filter_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
pool_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
ac = ['relu', 'relu', 'relu', 'relu', 'relu']
n_neurons_dense = 128

def main(config = None):
    run = wandb.init(config=config, resume=True)
    config = wandb.config

    run.name = "filters_" + str(config.filters) + "_dropout_" + str(config.dropout)

    num_filters = [config.filters] * 5
    
    if config.organisation == "double":
        for i in range(1, 5) : num_filters[i] = int(num_filters[i - 1] * 2)
    elif config.organisation == "half":
        for i in range(1, 5) : num_filters[i] = int(num_filters[i - 1] / 2)

    print(num_filters)

    # Model Definations
    model = Sequential()

    for i in range(5):
        model.add(Conv2D(num_filters[i], filter_size[i], input_shape = (h, w, d)))
        model.add(Activation(ac[i]))
        if config.batch_normalization == "yes" : model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = pool_size[i]))

    model.add(Flatten())
    model.add(Dense(n_neurons_dense, activation = 'relu'))
    if config.batch_normalization == "yes" : model.add(BatchNormalization())
    model.add(Dropout(config.dropout))
    model.add(Dense(n_labels, activation = 'softmax'))

    # Model Compilation
    model.compile(
        'adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
        )

    # Model Training
    if config.augmentation == "yes":
        model.fit(
            datagen.flow(train_x, to_categorical(train_y)),
            epochs = 28,
            callbacks = [WandbCallback()],
            validation_data = (val_x, to_categorical(val_y))
        )
    else:
        model.fit(
            train_x,
            to_categorical(train_y),
            epochs = 28,
            callbacks = [WandbCallback()],
            validation_data = (val_x, to_categorical(val_y))
        )
    
    run.finish()

sweep_config = {
  "name": "Sweep 4",

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
        "filters": {
            "values": [16, 32, 64]
        },
        "organisation" :{
            "values" : ["same", "double", "half"]
        },
        "augmentation": {
            "values": ["yes", "no"]
        },
        "dropout": {
            "values": [0.3, 0.5]
        },
        "batch_normalization": {
            "values": ["yes", "no"]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment 2")
wandb.agent(sweep_id, project="Assignment 2", function=main)