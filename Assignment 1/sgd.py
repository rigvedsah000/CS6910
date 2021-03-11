import wandb
import numpy as np
import  init_methods, forward_propagation, back_propagation, accuracy_loss

def sgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs = 100, eta = 0.1, init_strategy = "xavier", alpha = 0):

  print("Function Invoked: sgd")

  # Initialize params
  W, b =  init_methods.random_init(d, hl, ol) if init_strategy == "random" else init_methods.xavier_init(d, hl, ol)

  t, n_hl = 0, len(hl)

  while t < epochs:

    gW, gb = [], []

    for index, (x, y) in enumerate(zip(train_x, train_y)):

      # Forward propagation
      h, a = forward_propagation.forward_propagation(W, b, x, n_hl, ac)
    
      # Prediction (y hat)
      _y = h[n_hl + 1]
      
      # Backward propagation
      _gW, _gb = back_propagation.back_propagation(W, h, x, y, _y, n_hl, ac)

      if index == 0:
        gW = _gW
        gb = _gb
      else:
        gW = list(np.add(gW, _gW))
        gb = list(np.add(gb, _gb))

      # Update bias
      for index, (_b, _gb) in enumerate(zip(b, gb)):
        b[index] = _b - eta * np.array(_gb)

      # Update weights
      for index, (_W, _gW) in enumerate(zip(W, gW)):
        W[index] = _W - eta * (np.array(_gW) + alpha * _W)

    # Logging to WandB
      # val_acc, val_loss = accuracy_loss.get_accuracy_and_loss(W, b, val_x, val_y, n_hl, ac)
      # train_acc, train_loss = accuracy_loss.get_accuracy_and_loss(W, b, train_x, train_y, n_hl, ac)

      # wandb.log( { "val_accuracy": val_acc, "accuracy": train_acc, "val_loss": val_loss, "loss": train_loss } )

    t += 1

  return W, b
