import numpy as np

def logistic(x):
  res = []
  for _x in x:
     res.append( 1 / ( 1 + np.exp(-(float(_x))) ) )
  return np.array(res)

def softmax(x):
  res = []
  denom = sum([ np.exp(float(_x)) for _x in x ])
  for _x in x:
    num = np.exp(float(_x))
    res.append(num / denom)
  return np.array(res)

def tanh(x):
  return np.array([ np.tanh(_x) for _x in x ] )

def ReLU(x):
  return np.array( [max(0, _x) for _x in x] )
