from matplotlib import pyplot
from keras.datasets import fashion_mnist
import preprocessing, train
import accuracy_loss, init_strategy

 # Load dataset
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# Summarize loaded dataset
print("Train: X = ", train_X.shape, ", Y = ", train_Y.shape)
print("Test: X = ", test_X.shape, ", Y = ", test_Y.shape)

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Initialize an array
img = [-1]*10
count = 0

# Save indices of images corresponding to all different labels
for index, label in enumerate(train_Y):
  if img[label] == -1:
    img[label] = index
    count += 1
  
  if count == 10:
    break

# Plotting images correpsonding to different labels
fig = pyplot.figure()
fig.set_figwidth(15)
fig.set_figheight(15)

for index, image_index in enumerate(img):
  fig.add_subplot(5, 5, index + 1)
  pyplot.imshow(train_X[image_index], cmap='gray')

pyplot.show()


(N, w, h), n_labels = train_X.shape, len(labels)

# Number of datapoints
n = 5

# Dimension of datapoints
d = w * h

(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(d, n_labels, train_X, train_Y, test_X, test_Y)

_hl = [5]                                                                     # Hidden layers
_ol = [len(train_y[0])]                                                       # Output layers

W, b = train.train(train_x[:n], train_y[:n], d, _hl, _ol)

a, l = accuracy_loss.get_accuracy_and_loss(W, b, train_x[:n], train_y[:n], len(_hl))

print("Accuracy: ", a, " Loss: ", l)