import numpy as np
import functions

def forward_propagation(W, b, x, n_hl, ac):
  h, a = [ [] ], [ [] ]

  _h, _a = [], []

  for i in range(1, n_hl + 1):
    if i == 1:
      _a = np.dot( W[i], x ) + b[i]
    else:
      _a = np.dot( W[i], h[i - 1] ) + b[i]

    if ac == "sig":
      _h = functions.logistic(_a)
    
    elif ac == "tanh":
      _h = functions.tanh(_a)
    
    elif ac == "relu":
      _h = functions.ReLU(_a)

    a.append(_a)
    h.append(_h)

  _a = np.dot( W[n_hl + 1],  h[n_hl] ) + b[n_hl + 1]
  _y = functions.softmax(_a - max(_a))

  a.append(_a)
  h.append(_y)

  return h, a
