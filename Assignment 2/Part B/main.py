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

train_data = load_data.load_train_images("train")
(h, w, d), n_labels, batch_size = train_data[0].shape, 10, 128

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

new_model.fit(
    train_set,
    steps_per_epoch = ceil((float) (train_set.n) / train_set.batch_size),
    epochs = 25,
    validation_data = val_set,
    validation_steps = ceil((float) (val_set.n) / val_set.batch_size)
)

new_model.save('my_model.h5')
