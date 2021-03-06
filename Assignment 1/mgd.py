'''                                     Momentum Gradient descent ~ Minibatch version         '''


import forward_propagation,back_propagation,init_strategy,accuracy_loss
import numpy as np
import math
import random

def mgd (train_x,train_y, d, hl,ol,batch_size): 

  # Initialize paramaters
  W, b =  init_strategy.random_init2(d, hl, ol)

  eta = 0.001
  gamma=0.9
  grad_W,grad_b =  init_strategy.random_init2(d, hl, ol)  
  prev_W,prev_b= init_strategy.random_init2(d,hl,ol)
  
  iteration = 0
  epochs = 100

  while iteration < epochs:
    num_points_seen = 0
    for loc,(x,y_true) in enumerate(zip(train_x,train_y)):
      num_points_seen += 1
      
      #Forward Propagation
      h,a = forward_propagation.forward_propagation (W,b,x,len(hl))

      # Prediction (y hat) .It will be the last element(np array) of the h list
      y_pred = h[len(hl)+1]
    
      # Backward Propagation 
      grad_W_element , grad_b_element = back_propagation.back_propagation(W,h,x,y_true,y_pred,len(hl))

      if loc == 0 or num_points_seen == 1:
        for i in range(len(grad_W)):
          grad_W[i]=grad_W_element[i]
          grad_b[i]=grad_b_element[i]
      else:
        for i in range(len(grad_W)):
          grad_W[i] += grad_W_element[i]
          grad_b[i] += grad_b_element[i]


      if num_points_seen == batch_size or loc == len(train_x)-1 :
          num_points_seen = 0
          # Updating of prev_W,prev_b, W and b
          if iteration == 0:
            for i in range(1,len(W)):
              W[i]=W[i]-eta*grad_W[i]
              b[i]=b[i]-eta*grad_b[i]
              prev_W[i]=eta*grad_W[i]
              prev_b[i]=eta*grad_b[i]
          else:
            for i in range(1,len(W)):
              prev_W[i] = np.multiply(gamma,prev_W[i])+eta*grad_W[i]
              prev_b[i] = np.multiply(gamma,prev_b[i])+eta*grad_b[i]

              W[i]=W[i]-prev_W[i]
              b[i]=b[i]-prev_b[i]
              
          grad_W,grad_b = init_strategy.random_init2(d,hl,ol)

    train_acc, train_loss =  accuracy_loss.get_accuracy_and_loss(W, b, train_x, train_y, len(hl))

    print("\n\niteration number ",iteration," Training  Accuracy: ", train_acc, " Training Loss: ", train_loss)

    iteration += 1
  return W,b