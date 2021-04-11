### WAND Run ###
import numpy as np
import wandb, gc
from math import ceil
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

import load_data

train_data = load_data.load_train_images("train")
(h, w, d), n_labels, batch_size, n_neurons_dense = train_data[0].shape, 10, 128, 128

train_datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale = 1./255,
    horizontal_flip = True,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = [0.5, 1.5]
)

test_datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale = 1./255
)

train_datagen.fit(train_data)
test_datagen.fit(train_data)

del train_data
gc.collect()

train_set = train_datagen.flow_from_directory("train", target_size = (h, w), batch_size = batch_size)
val_set = test_datagen.flow_from_directory("val", target_size = (h, w), batch_size = batch_size)
test_set = test_datagen.flow_from_directory("test", target_size = (h, w), batch_size = batch_size)

# Model Defination
num_filters = [64, 64, 64, 64, 64]
filter_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
pool_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
ac = ['relu', 'relu', 'relu', 'relu', 'relu']

def main(config = None):
    run = wandb.init(config = config)
    config = wandb.config

    run.name = "filters_" + str(config.filters) + "_dropout_" + str(config.dropout)

    num_filters = [config.filters] * 5
    
    if config.organisation == "double":
        for i in range(1, 5) : num_filters[i] = int(num_filters[i - 1] * 2)
    elif config.organisation == "half":
        for i in range(1, 5) : num_filters[i] = int(num_filters[i - 1] / 2)
    
    # Model Defination
    model = Sequential()

    for i in range(5):
        model.add(Conv2D(num_filters[i], filter_size[i], input_shape = (h, w, d)))
        model.add(Activation(ac[i]))
        if config.batch_normalization == "yes" : model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = pool_size[i]))
        if config.batch_normalization == "yes" : model.add(BatchNormalization())

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
    model.fit(
        train_set,
        steps_per_epoch = ceil((float) (train_set.n) / train_set.batch_size),
        epochs = 26,
        callbacks = [WandbCallback()],
        validation_data = val_set,
        validation_steps = ceil((float) (val_set.n) / val_set.batch_size)
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
            "values": [32, 64]
        },
        "organisation" :{
            "values" : ["same", "double", "half"]
        },
        "augmentation": {
            "values": ["yes"]
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