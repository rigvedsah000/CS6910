from matplotlib import pyplot
from keras.datasets import fashion_mnist

def load_data():
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

    return train_X, train_Y, test_X, test_Y, labels