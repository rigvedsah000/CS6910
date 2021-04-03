import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import load_data, preprocessing

# Load Data
train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(h, w, d), n_labels = train_X[0].shape, len(labels)

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(h, w, d, n_labels, train_X, train_Y, test_X, test_Y)

datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 20,
    shear_range = 5/w,
    width_shift_range = 5/w,
    height_shift_range = 5/h
)

datagen.fit(train_x)

# Model Defination
num_filters = [64, 64, 64, 64, 64]
filter_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
pool_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
ac = ['relu', 'relu', 'relu', 'relu', 'relu']
n_neurons_dense = 256

model = Sequential()

for i in range(5):
    model.add(Conv2D(num_filters[i], filter_size[i], input_shape = (h, w, d)))
    model.add(Activation(ac[i]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = pool_size[i]))
    model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(n_neurons_dense, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(n_labels, activation = 'softmax'))

# Model Compilation
model.compile(
    'adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

# Model Training
model.fit(
    datagen.flow(train_x, to_categorical(train_y)),
    epochs = 15,
    validation_data = (val_x, to_categorical(val_y))
)

model.save_weights('cnn.h5' )