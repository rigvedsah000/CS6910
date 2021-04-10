import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    if code == 0:
        return InceptionV3(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == 1:
        return InceptionResNetV2(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == 2:
       return ResNet50(input_shape = (h, w, d), weights = "imagenet", include_top = False)
    elif code == 3:
        return Xception(input_shape = (h, w, d), weights = "imagenet", include_top = False)

model = get_pre_trained_model(1, h, w, d)
model.trainable = False
new_model = Model(inputs = model.input, outputs = Dense(n_labels, activation = "softmax")(Flatten()(model.output)))

new_model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

train_datagen = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 40,
    shear_range = 0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2
)

val_datagen = ImageDataGenerator()

train_set = train_datagen.flow(train_x, to_categorical(train_y))
val_set = val_datagen.flow(val_x, to_categorical(val_y))

new_model.fit(
    train_set,
    epochs = 15,
    validation_data = val_set
)