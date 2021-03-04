import numpy as np
import  init_strategy, forward_propagation, back_propagation

def vgd(train_x, train_y, d, hl, ol):
  
  # Initialize params
  W, b =  init_strategy.xavier_init(d, hl, ol)

  t, n_hl, epochs, eta = 0, len(hl), 1000, 0.001

  while t < epochs:

    gW, gb = [], []

    for index, (x, y) in enumerate(zip(train_x, train_y)):

      # Forward propagation
      h, a = forward_propagation.forward_propagation(W, b, x, n_hl)
    
      # Prediction (y hat)
      _y = h[n_hl + 1]

      # Backward propagation
      _gW, _gb = back_propagation.back_propagation(W, h, x, y, _y, n_hl)

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
      W[index] = _W - eta * np.array(_gW)

    t += 1

  return W, b
