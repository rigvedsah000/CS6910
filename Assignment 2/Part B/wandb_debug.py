### WAND Debug ###
import numpy as np
import wandb
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
import load_data, preprocessing

# Load Data
train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(h, w, d), n_labels = train_X[0].shape, len(labels)

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(h, w, d, n_labels, train_X, train_Y, test_X, test_Y)

def get_pre_trained_model(code, h, w, d):
    if code == "inception_v3":
        return InceptionV3(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == "inception_resnet_v2":
        return InceptionResNetV2(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == "resnet_50":
       return ResNet50(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == "xception":
        return Xception(input_shape = (h, w, d), weights = "imagenet", include_top = False)

train_datagen = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2
)

test_datagen = ImageDataGenerator()

train_set = train_datagen.flow(train_x, to_categorical(train_y))
val_set = test_datagen.flow(val_x, to_categorical(val_y))
test_set = test_datagen.flow(test_x, to_categorical(test_y))

def mainDebug(config = None):
    run = wandb.init(config = config)
    config = wandb.config

    run.name = "model_" + str(config.model) + "_trainable_layers_" + str(config.trainable_layers)

    model = get_pre_trained_model(config.model, h, w, d)
    
    model.trainable = False
    
    new_model = Model(inputs = model.input, outputs = Dense(n_labels, activation = "softmax")(Flatten()(model.output)))

    print(new_model.summary())

    new_model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    new_model.fit(
        train_set,
        epochs = 10,
        callbacks = [WandbCallback()],
        validation_data = val_set,
    )

    if config.trainable_layers == 1:
        print("Training Layer(s): ", config.trainable_layers)

    else:

        if config.trainable_layers == -1:
            print("Training Layer(s): ", config.trainable_layers)
            model.trainable = True
        else:
            print("Training Layer(s): ", config.trainable_layers)
            for layer in model.layers[len(model.layers) - 12 : ]:
                layer.trainable = True
        
        print(new_model.summary())

        new_model.compile(
            optimizer = Adam(1e-5),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        new_model.fit(
           train_set,
           epochs = 10,
           callbacks = [WandbCallback()],
           validation_data = val_set
        )
    
    # new_model.predict(test_set, callbacks = [WandbCallback()])
    
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
            "values": ["inception_v3"]
        },
        "trainable_layers" : {
            "values" : [3]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Test 3")
wandb.agent(sweep_id, project = "Test 3", function = mainDebug, count = 1)