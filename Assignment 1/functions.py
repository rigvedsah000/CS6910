import numpy as np


#  Activation functions : logistic , tanh and Relu 

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

#  Derivative of logistic , tanh  and ReLU activation function.

def derivative_logistic(x):
  return x * (1 - x)

def derivative_tanh(v):
  return 1 - np.square(v)

def derivative_ReLU(v):
  return np.array([1 if v[i]>0 else 0 for i in range(len(v))])





# Gradient of loss with respect to activation
def grad_a_cross_entropy(y_true,y_pred):
  grad_a = -1 * (y_true - y_pred)
  return grad_a


def grad_a_squared_loss (y_true , y_pred) :
  _y_l = y_pred[np.argmax(y_true)]
  _ga = 2 * (_y_l - 1) * _y_l * ( y_true - y_pred )
  return _ga



