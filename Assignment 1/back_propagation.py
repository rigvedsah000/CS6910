import numpy as np
import functions

def back_propagation(W, h, x, y, _y, n_hl, ac, lf):
  gW, gb = [ [] ], [ [] ]
  
  if lf == "cross_entropy" :
    _ga = functions.grad_a_cross_entropy(y,_y)
  else:
    _ga = functions.grad_a_squared_loss(y,_y)
  

  for i in range(n_hl + 1, 1, -1):
    _gW = np.outer( _ga, h[i-1] )
    _gb = np.array(_ga)

    _gh = np.dot( np.transpose(W[i]), _ga )

    if ac == "sig":
      _ga = _gh * functions.derivative_logistic( h[i - 1] )
    
    elif ac == "tanh":
      _ga = _gh * functions.derivative_tanh( h[i - 1] )
    
    elif ac == "relu":
       _ga = _gh * functions.derivative_ReLU( h[i - 1] )

    gW.insert(1, _gW)
    gb.insert(1, _gb)

  _gW = np.outer( _ga, x )
  _gb = np.array(_ga)

  gW.insert(1, _gW)
  gb.insert(1, _gb)

  return gW, gb
