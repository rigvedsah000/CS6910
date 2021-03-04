import numpy as np
import random
import math

# Random initialization - 1 of the Weights and biases
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



# Random initialization- 2 of the Weights and biases
def random_init2(d, hl, ol):
  # Defining W and b matrix ,  W : Weight matrix  , b: bias matrix
  W = [[]]                                                       
  b = [[]]                                                       
 
  for i, n_neurons in enumerate(hl):
    b.append(np.array([  random.uniform(-0.5,0.5) for i in range( n_neurons)  ]))
    
    if i == 0:
      W.append(np.array([ np.array([ random.uniform(-0.5,0.5) for i in range( d) ])  for j in range(n_neurons)   ]))
    else:
      W.append(np.array([ np.array([ random.uniform(-0.5,0.5) for i in range( hl[i - 1]) ]) for j in range(n_neurons)  ]))

    if i == len(hl)-1:
      b.append(np.array([ random.uniform(-0.5,0.5) for m in range(ol[0])  ]))
      W.append(np.array([ np.array([ random.uniform(-0.5,0.5) for p in range( hl[-1]) ])  for m in range(ol[0]) ]))

  return W, b




# Xavier initialization  of the Weights and biases
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



