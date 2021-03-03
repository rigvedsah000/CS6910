import random
import numpy as np

def xavier_init(d, hl, ol):
  
  print("Function Invoked: xavier init")

  W = [ np.array([]) ]                                                       # Weights between layers
  b = [ np.array([]) ]                                                       # Bias of layers

  # Xavier initialization

  for index, num_neurons in enumerate(hl):
    b.append( np.random.randn(num_neurons) )
    
    if index == 0:
      W.append( np.random.randn( num_neurons, d ) * np.sqrt( 1 / d ) )
    else:
      W.append( np.random.randn( num_neurons, hl[index - 1] ) * np.sqrt( 1 / hl[index - 1] ) )

  for num_neurons in ol:
    b.append( np.random.randn( num_neurons ) )
    W.append( np.random.randn( num_neurons, hl[-1] ) * np.sqrt( 1 / hl[-1] ) )

  return W, b