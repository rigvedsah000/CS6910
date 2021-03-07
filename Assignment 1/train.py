import vgd, sgd

def train(train_x, train_y, val_x, val_y, d, hl, ol, config):

  print("Function Invoked: train")
  
  epochs, eta, init_strategy, optimiser, batch_size, ac =  config["epochs"], config["learning_rate"],  config["init_strategy"], config["optimiser"], config["batch_size"], config["ac"]

  if optimiser == "vgd":
    return vgd.vgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy)
  
  elif optimiser == "sgd":
    return sgd.sgd(train_x, train_y, val_x, val_y, d, hl, ol, ac, epochs, eta, init_strategy)

  # elif optimiser == "mgd":
  #   return mgd.mgd(train_x, train_y, hl, ol, ac, epochs, eta, init_strategy, batch_size)

  # elif optimiser == "nag":
  #   return nag.nag(train_x, train_y, hl, ol, ac, epochs, eta, init_strategy, batch_size)