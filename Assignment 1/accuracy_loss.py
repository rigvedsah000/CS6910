import numpy as np
import functions

def predict(W, b, x, n_hl, ac):
  a,h=[],[]
  for i in range(1, n_hl + 1):
    if i == 1:
      a =  np.dot(W[i],x) + b[i] 
    else:
      a =  np.dot(W[i],h) + b[i]
    
    if ac == "sig":
      h = functions.logistic(a)
    
    elif ac == "tanh":
      h = functions.tanh(a)
    
    elif ac == "relu":
      h = functions.ReLU(a)
  
  a = np.dot(W[n_hl+1],h ) + b[n_hl+1]
  y_pred = functions.softmax(a-max(a))

  return y_pred



#  Function to compute accuracy and loss
def get_accuracy_and_loss(W, b, data_x, data_y, hl, ac, lf):
  correct, loss = 0, 0

  for i in range(len(data_x)):
    y_pred = predict(W, b, data_x[i], hl, ac)

    y_true = data_y[i]

    index_1 = np.argmax(y_true)
    index_2 = np.argmax(y_pred)
    if y_true[index_2] == 1:
      correct += 1

    if lf == "cross_entropy":
      loss += (-np.log(y_pred[index_1] )  )
    else:
      loss +=  np.dot( y_pred - y_true , y_pred - y_true )

  accuracy =  correct /len(data_x)
  average_loss = loss /len(data_x)

  return accuracy, average_loss




def get_accuracy_loss_and_prediction(W, b, data_x, data_y, hl, ac, lf):
  correct, loss, y, _y = 0, 0, [], []

  for i in range(len(data_x)):
    y_pred = predict(W, b, data_x[i], hl, ac)

    y_true = data_y[i]

    index_1 = np.argmax(y_true)
    index_2 = np.argmax(y_pred)

    y.append(index_1)
    _y.append(index_2)

    if y_true[index_2] == 1:
      correct += 1

    if lf == "cross_entropy":
      loss += (-np.log(y_pred[index_1] )  )
    else:
      loss +=  np.dot( y_pred - y_true , y_pred - y_true )

  accuracy =  correct /len(data_x)
  average_loss = loss /len(data_x)

  return accuracy, average_loss, y, _y