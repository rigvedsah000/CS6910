import numpy as np
import functions

def predict(W, b, x, n_hl):
  a,h=[],[]
  for i in range(1, n_hl + 1):
    if i == 1:
      a =  np.dot(W[i],x) + b[i] 
    else:
      a =  np.dot(W[i],h) + b[i]
    
    h = functions.logistic(a)
  
  a = np.dot(W[n_hl+1],h ) + b[n_hl+1]
  y_pred = functions.softmax(a-max(a))

  return y_pred




def get_accuracy_and_loss(W, b, data_x, data_y, hl):
  correct, loss = 0, 0

  for i in range(len(data_x)):
    y_pred = predict(W, b, data_x[i], hl)

    y_true = data_y[i]

    index_1 = np.argmax(y_true)
    index_2 = np.argmax(y_pred)
    if y_true[index_2] == 1:
      correct += 1

    loss += (-np.log(y_pred[index_1] )  )


  return correct /len(data_x), loss / len(data_x)


