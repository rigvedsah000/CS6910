import random
import numpy as np

def random_init(d, hl, ol):
  
  print("Function Invoked: random init")
  
  W = [ np.array([]) ]                                                       # Weights between layers
  b = [ np.array([]) ]                                                       # Bias of layers

  # Random initialization

  for index, num_neurons in enumerate(hl):
    b.append( np.random.randn(num_neurons) )
    
    if index == 0:
      W.append( np.random.randn( num_neurons, d ) )
    else:
      W.append( np.random.randn( num_neurons, hl[index - 1] )  )

  for num_neurons in ol:
    b.append( np.random.randn( num_neurons ) )
    W.append( np.random.randn( num_neurons, hl[-1] ) )

  return W, b