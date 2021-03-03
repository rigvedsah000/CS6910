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

def forward_propagation(W, b, x, n_hl):
  h, a = [ [] ], [ [] ]

  _h, _a = [], []

  for i in range(1, n_hl + 1):
    if i == 1:
      _a = np.dot( W[i], x ) + b[i]
    else:
      _a = np.dot( W[i], h[i - 1] ) + b[i]

    _h = logistic(_a)

    a.append(_a)
    h.append(_h)

  _a = np.dot( W[n_hl + 1],  h[n_hl] ) + b[n_hl + 1]
  _y = softmax(_a - max(_a))

  a.append(_a)
  h.append(_y)

  return h, a