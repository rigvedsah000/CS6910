import cv2, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import keras
import os
import random

def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast( x > 0, "float32") * dy
    return tf.nn.relu(x), grad
     
def guided_backpropagate(model, layer, i, j, k, x, y, z, img, size):
    gb_model = Model(
        inputs  = [model.inputs],
        outputs = [model.layers[layer].output]
    )

    for layer in gb_model.layers: 
        if "activation" in layer.name : layer.activation = guidedRelu

    with tf.GradientTape() as tape:
        input = tf.cast(img, tf.float32)
        tape.watch(input)
        
        multiplier = np.zeros((1, x, y, z))
        
        multiplier[0][i][j][k] = 1
        
        output = gb_model(input) * multiplier

    grads = tape.gradient(output, input)[0]
    
    return cv2.resize(np.asarray(grads), size)
    
def process_image(img):
    img = img.copy()
    img -= img.mean()
    img /= (img.std() + K.epsilon())
    img *= 0.25
    
    img = np.clip(img, 0, 255).astype("uint8")
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


files = os.listdir("train/")
folder_selected = files[random.randrange(0,len(files))]
snaps = os.listdir("train/"+folder_selected)
snap_selected = snaps[random.randrange(0,len(snaps))]

img = cv2.imread("train/"+folder_selected+"/"+snap_selected)

model = keras.models.load_model('my_model.h5')

activation_index = 0
all_layers = model.layers
for i in range(len(all_layers)):
    if "activation" in all_layers[i].name:
        activation_index += 1
        if activation_index == 5:
            activation_index = i
            break

req_shape = model.layers[activation_index].output_shape
x,y,z = req_shape[1],req_shape[2],req_shape[3]

fig = plt.figure(figsize=(20, 2.5))
plt.axis('off') 

item = 1

for pl in range(10):
    i,j,k = random.randrange(0, x),random.randrange(0, y),random.randrange(0, z)
    gb_img = guided_backpropagate(model, activation_index, i, j, k, x, y, z, np.expand_dims(img, axis = 0), (100, 100))
    gb_img = process_image(gb_img)

    ax=fig.add_subplot(1,10,item)
    ax.imshow(gb_img)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_title("Feature map :"+str(k)+"\n Neuron no: ("+str(i)+","+str(j)+")")
    item+=1

plt.show()
plt.savefig("Guided Backprop")