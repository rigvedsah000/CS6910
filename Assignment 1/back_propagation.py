import numpy as np

def derivative_logistic(x):
  return x * (1 - x)

def back_propagation(W, h, x, y, _y, n_hl):
  gW, gb = [ [] ], [ [] ]
  
  _ga = -1 * (y - _y)

  for i in range(n_hl + 1, 1, -1):
    _gW = np.outer( _ga, h[i-1] )
    _gb = np.array(_ga)

    _gh = np.dot( np.transpose(W[i]), _ga )

    _ga = _gh * derivative_logistic( h[i - 1] )

    gW.insert(1, _gW)
    gb.insert(1, _gb)

  _gW = np.outer( _ga, x )
  _gb = np.array(_ga)

  gW.insert(1, _gW)
  gb.insert(1, _gb)

  return gW, gb
