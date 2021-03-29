import os, os.path, cv2, numpy as np

def load_data():

    path_train, path_test, train_X, train_Y, test_X, test_Y, labels = "train", "test", [], [], [], [], []

    # Load Training Data
    for i, d in enumerate(os.listdir(path_train)):
        
        _dir = os.path.join(path_train, d)
    
        if os.path.isdir(_dir):

            labels.append(d)

            print("Processing Folder: ", d)
        
            for f in os.listdir(_dir):

                if f == ".DS_Store":
                    continue
                
                train_X.append(np.array(cv2.imread(os.path.join(_dir, f))))
                train_Y.append(i)

    # Load Test Data
    for i, d in enumerate(os.listdir(path_test)):
        
        _dir = os.path.join(path_test, d)
    
        if os.path.isdir(_dir):

            print("Processing Folder: ", d)
        
            for f in os.listdir(_dir):

                if f == ".DS_Store":
                    continue
                
                test_X.append(np.array(cv2.imread(os.path.join(_dir, f))))
                test_Y.append(i)

    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y), labels