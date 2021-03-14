from keras.datasets import fashion_mnist

def load_data():
    # Load dataset
    (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

    # Summarize loaded dataset
    print("Train: X = ", train_X.shape, ", Y = ", train_Y.shape)
    print("Test: X = ", test_X.shape, ", Y = ", test_Y.shape)

    labels = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

    return train_X, train_Y, test_X, test_Y, labels
    