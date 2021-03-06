'''                                     RMSPROP Gradient descent ~ Minibatch version                                    '''

import forward_propagation,back_propagation,init_methods,accuracy_loss
import numpy as np
import math
import random
import wandb


def rmsprop(train_x, train_y, val_x, val_y, d, hl, ol, ac, lf, epochs, eta, init_strategy,batch_size):

    # Initialize paramaters
    if  init_strategy  ==  "random" :
        W, b = init_methods.random_init2(d, hl, ol)
    else :
        W, b = init_methods.xavier_init(d, hl, ol)         

    hist_W,hist_b =  init_methods.random_init2(d, hl, ol)
    grad_W,grad_b =  init_methods.random_init2(d,hl,ol)

    epsilon,beta1 = 1e-8 , 0.95
    iteration = 0
  
    while iteration < epochs :
        num_points_seen = 0

        for loc, (x, y_true) in enumerate(zip(train_x, train_y)):
            num_points_seen += 1
            # Forward propagation
            h,a = forward_propagation.forward_propagation(W,b,x,len(hl),ac)
            
            # Prediction (y hat) . It will be the last element(np array) of the h list
            y_pred = h[len(hl)+1]

            # Backward propagation
            grad_W_element, grad_b_element = back_propagation.back_propagation(W,h,x,y_true,y_pred,len(hl),ac, lf)

            if loc == 0 or num_points_seen==1:
                for i in range(len(grad_W)):
                    grad_W[i] = grad_W_element[i]
                    grad_b[i] = grad_b_element[i]
            else:
                for i in range(len(grad_W)):
                    grad_W[i] += grad_W_element[i]
                    grad_b[i] += grad_b_element[i]

            if num_points_seen == batch_size or loc == len(train_x)-1:
                num_points_seen = 0

                if iteration == 0:
                    for i in range(1,len(W)):
                        hist_W[i] = (1-beta1)* np.square(grad_W[i])
                        hist_b[i] = (1-beta1)*np.square(grad_b[i])
                        W[i] = W[i] - (eta / np.sqrt(hist_W[i] + epsilon))*grad_W[i]
                        b[i] = b[i] - (eta/np.sqrt(hist_b[i] +epsilon )) *grad_b[i]
                else:
                    for i in range(1,len(W)):
                        hist_W[i] = beta1*hist_W[i] + (1-beta1)* np.square(grad_W[i])
                        hist_b[i] = beta1* hist_b[i] +(1-beta1)*  np.square(grad_b[i])
                        W[i] = W[i] - (eta / np.sqrt(hist_W[i] + epsilon))*grad_W[i]
                        b[i] = b[i] - (eta/np.sqrt(hist_b[i] +epsilon )) *grad_b[i]
                
                grad_W,grad_b=init_methods.random_init2(d,hl,ol)
        
        if lf == "cross_entropy":
            train_acc, train_loss = accuracy_loss.get_accuracy_and_loss(W, b, train_x, train_y, len(hl), ac, lf)
            val_acc, val_loss = accuracy_loss.get_accuracy_and_loss(W, b, val_x, val_y, len(hl), ac, lf)
            wandb.log( { "val_accuracy": val_acc, "accuracy": train_acc, "val_loss": val_loss, "loss": train_loss } )

        # print("\n\niteration number ",iteration," Training  Accuracy: ", train_acc, " Training Loss: ", train_loss)
        # print("\n\niteration number ",iteration," validation  Accuracy: ", val_acc, " validation Loss: ", val_loss)
        
        iteration += 1
    return W, b
