import wandb
import numpy as np
import  init_methods, forward_propagation, back_propagation, accuracy_loss

def update_nadam_Wb(t, index, count, beta1, beta2, epsilon, eta, n_hl, batch_size, m_W, m_b, v_W, v_b, gW, gb, W, b, ac):
    for i in range(1, n_hl + 2):
      if t == 0 and (index + 1) == batch_size:
        m_W[i] = (1 - beta1) * gW[i]
        m_b[i] = (1 - beta1) * gb[i]

        v_W[i] = (1 - beta2) * np.square(gW[i])
        v_b[i] = (1 - beta2) * np.square(gb[i])
      else:
        m_W[i] = beta1 * m_W[i] + (1 - beta1) * gW[i]
        m_b[i] = beta1 * m_b[i] + (1 - beta1) * gb[i]

        v_W[i] = beta2 * v_W[i] + (1 - beta2) * np.square(gW[i])
        v_b[i] = beta2 * v_b[i] + (1 - beta2) * np.square(gb[i])

    # Update bias
    for index, (_b, _gb, _m_b, _v_b) in enumerate(zip(b, gb, m_b, v_b)):
      
      _m_b_hat = _m_b / (1 - np.power( beta1, count ))
      _v_b_hat = _v_b / (1 - np.power( beta2, count ))
      
      b[index] = _b - (eta / np.sqrt(_v_b_hat + epsilon)) * _m_b_hat

    # Update weights
    for index, (_W, _gW, _m_W, _v_W) in enumerate(zip(W, gW, m_W, v_W)):

      _m_W_hat = _m_W / (1 - np.power( beta1, count ))
      _v_W_hat = _v_W / (1 - np.power( beta2, count ))

      W[index] = _W - (eta / np.sqrt(_v_W_hat + epsilon)) * _m_W_hat

def nadam(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs = 100, eta = 0.1, init_strategy = "xavier", batch_size = 1):

  print("Function Invoked: nadam")
  
  # Initialize params
  W, b =  init_methods.random_init(d, hl, ol) if init_strategy == "random" else init_methods.xavier_init(d, hl, ol)

  n_hl = len(hl)

  t, beta1, beta2, epsilon, count = 0, 0.9, 0.999, 1e-8, 0
  v_W, v_b, m_W, m_b = [ np.array([]) ] * (n_hl + 2), [ np.array([]) ] * (n_hl + 2), [ np.array([]) ] * (n_hl + 2), [ np.array([]) ] * (n_hl + 2)
  
  while t < epochs:

    gW, gb, W_look_ahead, b_look_ahead = [], [], [ np.array([]) ] * (n_hl + 2), [ np.array([]) ] * (n_hl + 2)

    for index, (x, y) in enumerate(zip(train_x, train_y)):

      if index % batch_size == 0:

        if t == 0 and index == 0:
          W_look_ahead = np.copy(W)
          b_look_ahead = np.copy(b)
      
        else:
          for _index, (_b, _m_b, _v_b) in enumerate(zip(b, m_b, v_b)):
            _m_b_hat = ( beta1 * _m_b ) / (1 - np.power( beta1, count + 1 ))
            _v_b_hat = ( beta2 * _v_b ) / (1 - np.power( beta2, count + 1 ))
            b_look_ahead[_index] = _b - (eta / np.sqrt(_v_b_hat + epsilon)) * _m_b_hat

          for _index, (_W, _m_W, _v_W) in enumerate(zip(W, m_W, v_W)):
            _m_W_hat = ( beta1 * _m_W ) / (1 - np.power( beta1, count + 1 ))
            _v_W_hat = ( beta2 * _v_W ) / (1 - np.power( beta2, count + 1 ))
            W_look_ahead[_index] = _W - (eta / np.sqrt(_v_W_hat + epsilon)) * _m_W_hat
            
      # Forward propagation
      h, a = forward_propagation.forward_propagation(W_look_ahead, b_look_ahead, x, n_hl, ac)
    
      # Prediction (y hat)
      _y = h[n_hl + 1]

      # Backward propagation
      _gW, _gb = back_propagation.back_propagation(W_look_ahead, h, x, y, _y, n_hl, ac)

      if index % batch_size == 0:
        gW = _gW
        gb = _gb
      else:
        gW = np.add(gW, _gW)
        gb = np.add(gb, _gb)


      if (index + 1) % batch_size == 0:
        count += 1
        update_nadam_Wb(t, index, count, beta1, beta2, epsilon, eta, n_hl, batch_size, m_W, m_b, v_W, v_b, gW, gb, W, b, ac)
        gW, gb, W_look_ahead, b_look_ahead = [], [], [ np.array([]) ] * (n_hl + 2), [ np.array([]) ] * (n_hl + 2)

    if len(train_x) % batch_size != 0:
      count += 1
      index = batch_size - 1 if len(train_x) < batch_size else -1
      update_nadam_Wb(t, index, count, beta1, beta2, epsilon, eta, n_hl, batch_size, m_W, m_b, v_W, v_b, gW, gb, W, b, ac)

    # Logging to WandB
    val_acc, val_loss = accuracy_loss.get_accuracy_and_loss(W, b, val_x, val_y, n_hl, ac)
    train_acc, train_loss = accuracy_loss.get_accuracy_and_loss(W, b, train_x, train_y, n_hl, ac)

    wandb.log( { "val_accuracy": val_acc, "accuracy": train_acc, "val_loss": val_loss, "loss": train_loss } )

    t += 1

  return W, b