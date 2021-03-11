import vgd, sgd, mgd, nag, adam , rmsprop, nadam

def train(train_x, train_y, val_x, val_y, d, hl, ol, config):

  print("Function Invoked: train")
  
  epochs, eta, alpha, init_strategy, optimiser, batch_size, ac =  config["epochs"], config["learning_rate"], config["weight_decay"],  config["init_strategy"], config["optimiser"], config["batch_size"], config["ac"]

  if optimiser == "vgd":
    return vgd.vgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy, alpha)
  
  elif optimiser == "sgd":
    return sgd.sgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy, alpha)

  elif optimiser == "mgd":
    return mgd.mgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy,batch_size, alpha)

  elif optimiser == "nag":
    return nag.nag(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy,batch_size, alpha)

  elif optimiser == "adam":
    return adam.adam(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy, batch_size)

  elif optimiser == "rmsprop":
    return rmsprop.rmsprop(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy, batch_size)

  elif optimiser == "nadam":
    return nadam.nadam(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy, batch_size)