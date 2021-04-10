import numpy as np
from math import ceil

# Function to distribute the training dataset into training and validation set
def get_validation_data (dataset, label) :
  n_class=np.unique(label, return_counts=True)[1]
  v_class = [0]*len(n_class)
 
  validation_x=[]
  validation_y=[]
  training_x=[]
  training_y=[]

  shuffler=np.random.permutation(len(dataset))
  shuffled_dataset = dataset[shuffler]
  shuffled_label = label[shuffler]

  index=0
  validation_count=ceil(0.1*len(dataset))
  while (validation_count>0) :
      if index==len(dataset):
        break
      if ( v_class[shuffled_label[index]] < 0.1*n_class[shuffled_label[index]]):
        validation_x.append(shuffled_dataset[index])
        validation_y.append(shuffled_label[index])
        v_class[shuffled_label[index]] += 1
        validation_count -= 1
      else:
        training_x.append(shuffled_dataset[index])
        training_y.append(shuffled_label[index])
      index += 1
      
  for i in range(index,len(dataset)):
      training_x.append(shuffled_dataset[i])
      training_y.append(shuffled_label[i])
  

  return (np.array(training_x),np.array(training_y)),(np.array(validation_x),np.array(validation_y))

# Function to normalize 
def normalize(train_x, val_x, test_x, h, w, d):
  
  print("Function Invoked: normalize")
  
  mean_x = np.zeros((h, w, d))

  for x in train_x : mean_x += x
  mean_x =  mean_x / len(train_x)

  normalized_train_x = (train_x - mean_x) / 255 
  normalized_val_x = (val_x - mean_x) / 255 
  normalized_test_x = (test_x - mean_x) / 255 

  return normalized_train_x, normalized_val_x, normalized_test_x

def pre_process(h, w, d, n_labels, train_X, train_Y, test_X, test_Y):
  
  print("Function Invoked: pre-process")

  (train_x, train_y), (val_x, val_y) = get_validation_data(train_X, train_Y)

  train_x, val_x, test_x = normalize(train_x, val_x, test_X, h, w, d)

  return (train_x, train_y), (val_x, val_y), (test_x, test_Y)
