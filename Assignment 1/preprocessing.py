import numpy as np

# Function to flatten the images.
def flatten (train , test) :
  flattened_train = np.array([train[i].flatten() for i in range(len(train))])
  flattened_test  = np.array([test[i].flatten() for i in range(len(test))])
  return flattened_train , flattened_test





# Function to distribute the training dataset into training and validation set
def get_validation_data (dataset, label) :
  n_class=np.unique(label,return_counts=True)[1]
  v_class = [0]*len(n_class)
 
  validation_x=[]
  validation_y=[]
  training_x=[]
  training_y=[]

  shuffler=np.random.permutation(len(dataset))
  shuffled_dataset = dataset[shuffler]
  shuffled_label = label[shuffler]

  index=0
  validation_count=int(0.1*len(dataset))
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




# Function to expand the labels 
def expandLabels(train_y, val_y, test_y, n_classes):
  exp_train_y = np.array([ [0]*n_classes for i in range(len(train_y)) ])
  exp_val_y = np.array([ [0]*n_classes for i in range(len(val_y)) ])
  exp_test_y = np.array([[0]*n_classes for i in range(len(test_y)) ])

  for i in range(len(train_y)):
    true_label = train_y[i]
    exp_train_y[i][true_label] = 1

  for i in range(len(val_y)):
    true_label = val_y[i]
    exp_val_y[i][true_label] = 1
  
  for i in range(len(test_y)):
    true_label = test_y[i]
    exp_test_y[i][true_label] = 1

  return exp_train_y , exp_val_y, exp_test_y



# Function to normalize 
def normalize(train_x, val_x, test_x, d):
  print("Function Invoked: normalize")
  mean_x = [0]*d
  for row in train_x : mean_x += row
  mean_x =  mean_x / len(train_x)

  normalized_train_x = (train_x - mean_x)/255 
  normalized_val_x = (val_x - mean_x)/255 
  normalized_test_x = (test_x - mean_x)/255 

  return normalized_train_x, normalized_val_x, normalized_test_x





# Preprocessing of the data : includes Flattening of the images(2-d matrix) , distributing training into training and validation data , normlizing 
# and expanding labels of the y values .

def pre_process(d, n_labels, train_x, train_y, test_x, test_y):
  print("Function Invoked: pre-process")
  flat_train_x, flat_test_x = flatten(train_x, test_x)   # flat_train_x is a mixture of test and validation data 

  (train_x, unexpanded_train_y), (val_x, unexpanded_val_y) = get_validation_data(flat_train_x, train_y)

  train_x, val_x, test_x = normalize(train_x, val_x, flat_test_x, d)
  train_y, val_y, test_y = expandLabels(unexpanded_train_y, unexpanded_val_y, test_y, n_labels)

  return (train_x, train_y), (val_x, val_y), (test_x, test_y)
