# Visualising Filters
from matplotlib import pyplot as plt
import cv2, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.models.load_model('my_model.h5')

first_convlayer_index = 0
all_layers = model.layers
for i in range(len(all_layers)):
    if "conv" in all_layers[i].name:
        first_convlayer_index = i
        break

first_layer = model.layers[first_convlayer_index]
filters = first_layer.get_weights()[0]

# normalizing filters
filters = (filters - filters.min()) / (filters.max() - filters.min())

# plotting the filters
n_filters = filters.shape[3]
rows,cols = np.sqrt(n_filters),np.sqrt(n_filters)

fig = plt.figure(figsize=(rows*1, cols*1))
plt.axis('off') 
plt.title("Filter visualisation of first convolutional layer")
item=1

for i in range(n_filters):
    cur_filter = filters[:, :, :, i]

    ax = fig.add_subplot(rows, cols, item)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.imshow(cur_filter)
    item += 1

plt.show()
plt.savefig('filter_visualise.png')