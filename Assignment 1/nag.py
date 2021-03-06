'''                                     Nesterov Gradient descent ~ Minibatch version                    '''

import forward_propagation,back_propagation,init_strategy,accuracy_loss
import numpy as np
import math
import random


def nag(train_x,train_y, d, hl,ol,batch_size):  

    # Initialize paramaters
    W, b =  init_strategy.random_init2(d, hl, ol)

    grad_W, grad_b =   init_strategy.random_init2(d, hl, ol)

    iteration = 0 
    gamma=0.9
    epochs = 100
    eta = 0.01

    prev_W,prev_b =  init_strategy.random_init2(d,hl,ol)
    W_lookahead,b_lookahead = init_strategy.random_init2(d,hl,ol)
  
    while iteration < epochs:
        num_points_seen = 0
        # grad_W, grad_b = [],[]


        for loc,(x,y_true) in enumerate(zip(train_x,train_y)):
            num_points_seen += 1

            if num_points_seen == 1:
                if iteration == 0 :
                    for i in range(1,len(W)):
                        W_lookahead[i]=W[i]-gamma*prev_W[i]
                        b_lookahead[i]=b[i]-gamma*prev_b[i]
                else:
                    for i in range(1,len(W)):
                        W_lookahead[i]=W[i]
                        b_lookahead[i]=b[i]

            #Forward Propagation
            h , a = forward_propagation.forward_propagation (W_lookahead,b_lookahead,x,len(hl))

            # Prediction (y hat) . It will be the last element(np array) of the h list
            y_pred = h[len(hl) +1]

            # Backward Propagation 
            grad_W_element , grad_b_element = back_propagation.back_propagation(W_lookahead,h,x,y_true,y_pred,len(hl))

            if loc == 0 or num_points_seen==1:
                for i in range(len(grad_W)):
                    grad_W[i] = grad_W_element[i]
                    grad_b[i] = grad_b_element[i]  
            else:
                for i in range(len(grad_W)):
                    grad_W[i] += grad_W_element[i]
                    grad_b[i] += grad_b_element[i]

            if num_points_seen == batch_size or loc == len(train_x)-1 :
                num_points_seen = 0

                if iteration == 0:
                    for i in range(1,len(W)):
                        W[i]=W[i]-eta*grad_W[i]
                        b[i]=b[i]-eta*grad_b[i]
                        prev_W[i]=eta*grad_W[i]
                        prev_b[i]=eta*grad_b[i]
                else:
                    for i in range(1,len(W)):
                        prev_W[i] = gamma*prev_W[i]+eta*grad_W[i]
                        prev_b[i] = gamma*prev_b[i]+eta*grad_b[i]

                        W[i]=W[i]-prev_W[i]
                        b[i]=b[i]-prev_b[i]


                grad_W, grad_b =  init_strategy.random_init2(d,hl,ol)


        train_acc, train_loss =  accuracy_loss.get_accuracy_and_loss(W, b, train_x, train_y, len(hl))


        print("\n\niteration number ",iteration," Training  Accuracy: ", train_acc, " Training Loss: ", train_loss)

        iteration += 1
        return W,b