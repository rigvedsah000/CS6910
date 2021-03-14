import load_data, preprocessing, train, accuracy_loss, plot

train_X, train_Y, test_X, test_Y, labels = load_data.load_data()

(N, w, h), n_labels = train_X.shape, len(labels)

# Number of datapoints to train
n = 1000

# Dimension of datapoints
d = w * h

config = {
    "learning_rate" : 0.001,
    "epochs": 100,
    "optimiser": "mgd",
    "hidden_layers": 3,
    "hidden_layer_size": 32,
    "ac": "tanh",
    "batch_size": 32,
    "init_strategy": "xavier",
    "weight_decay": 0.0005
}

# Data Preprocessing
# (train_x, train_y), (val_x, val_y), (test_x, test_y) = preprocessing.pre_process(d, n_labels, train_X, train_Y, test_X, test_Y)

# hl = [config["hidden_layer_size"]] * config["hidden_layers"]                 # Hidden layers
# ol = [len(train_y[0])]                                                       # Output layer

# # Model Training
# W, b = train.train(train_x[:n], train_y[:n], val_x[:n], val_y[:n], d, hl, ol, config)

# # Checking Accuracy
# a, l = accuracy_loss.get_accuracy_and_loss(W, b, train_x[:n], train_y[:n], len(hl), config["ac"])

# print("Accuracy: ", a, " Loss: ", l)

plot.confusion_matrix(labels, [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 7])