### WAND Run ###
import wandb, numpy as np
from math import ceil
from wandb.keras import WandbCallback
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Model Libs
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception

#############################
# Set Parameters Here
############################
(h, w, d), n_labels, batch_size = (100, 100, 3), 10, 128

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = [0.5, 1.5]
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

train_set = train_datagen.flow_from_directory("train", target_size = (h, w), batch_size = batch_size)
val_set = test_datagen.flow_from_directory("val", target_size = (h, w), batch_size = batch_size)
test_set = test_datagen.flow_from_directory("test", target_size = (h, w), batch_size = batch_size)


def get_pre_trained_model(code, h, w, d):
    if code == "inception_v3":
        return InceptionV3(input_shape = (h, w, d), weights = "imagenet", include_top = False), [12, 17]
    elif code == "inception_resnet_v2":
        return InceptionResNetV2(input_shape = (h, w, d), weights = "imagenet", include_top = False), [2, 4]
    elif code == "resnet_50":
       return ResNet50(input_shape = (h, w, d), weights = "imagenet", include_top = False), [4, 7]
    elif code == "xception":
        return Xception(input_shape = (h, w, d), weights = "imagenet", include_top = False), [3, 6]

def main(config = None):
    run = wandb.init(config = config)
    config = wandb.config

    run.name = "model_" + str(config.model) + "_trainable_conv_layers_" + str(config.trainable_conv_layers)

    model, conv_layers = get_pre_trained_model(config.model, h, w, d)
    
    model.trainable = False
    
    new_model = Model(inputs = model.input, outputs = Dense(n_labels, activation = "softmax")(Flatten()(model.output)))

    new_model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    new_model.fit(
        train_set,
        steps_per_epoch = ceil((float) (train_set.n) / train_set.batch_size),
        epochs = 25,
        callbacks = [WandbCallback()],
        validation_data = val_set,
        validation_steps = ceil((float) (val_set.n) / val_set.batch_size)
        )

    if config.trainable_conv_layers == 0:
        print("")
    else:

        if config.trainable_conv_layers == -1:
            model.trainable = True
        else:
            for layer in model.layers[len(model.layers) - conv_layers[config.trainable_conv_layers - 1] : ]:
                layer.trainable = True

        new_model.compile(
            optimizer = Adam(1e-5),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        new_model.fit(
            train_set,
            steps_per_epoch = ceil((float) (train_set.n) / train_set.batch_size),
            epochs = 25,
            callbacks = [WandbCallback()],
            validation_data = val_set,
            validation_steps = ceil((float) (val_set.n) / val_set.batch_size)
            )

    #new_model.save('my_model.h5')
       
    run.finish()

sweep_config = {
  "name": "Part B Sweep 1",

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
        "model": {
            "values": ["inception_v3", "inception_resnet_v2", "resnet_50", "xception"]
        },
        "trainable_conv_layers" : {
            "values" : [0, 1, 2, -1]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment 2")
wandb.agent(sweep_id, project="Assignment 2", function=main)
