import load_data, preprocessing, train, accuracy_loss

train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(N, w, h), n_labels = train_X.shape, len(labels)

# Number of datapoints to train
n = 100

# Dimension of datapoints
d = w * h

config = {
    "learning_rate" : 0.0001,
    "epochs": 100,
    "optimiser": "sgd",
    "hidden_layers": 3,
    "hidden_layer_size": 5,
    "ac": "relu",
    "batch_size": 32,
    "init_strategy": "random"
}

# Data Preprocessing
(train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(d, n_labels, train_X, train_Y, test_X, test_Y)

hl = [config["hidden_layer_size"]] * config["hidden_layers"]               # Hidden layers
ol = [len(train_y[0])]                                                       # Output layer

# Model Training
W, b = train.train(train_x[:n], train_y[:n], val_x[:n], val_y[:n], d, hl, ol, config)

# Checking Accuracy
a, l = accuracy_loss.get_accuracy_and_loss(W, b, train_x[:n], train_y[:n], len(hl), config["ac"])

print("Accuracy: ", a, " Loss: ", l)