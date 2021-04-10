import os, os.path, cv2

def resize_train(height, width):
    path = "inaturalist_12K/train"

    for d in os.listdir(path):
        
        _dir = os.path.join(path, d)
    
        if os.path.isdir(_dir):
            os.mkdir(os.path.join("train", d))
            
            print("Processing Folder: ", d)
            
            for i, f in enumerate(os.listdir(_dir)):

                if f == ".DS_Store":
                    continue

                img = cv2.imread(os.path.join(_dir, f))
                resized = cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join("train", d, f), resized)


def resize_test(height, width):
    path = "inaturalist_12K/val"

    for d in os.listdir(path):
        
        _dir = os.path.join(path, d)
    
        if os.path.isdir(_dir):
            os.mkdir(os.path.join("test", d))
            
            print("Processing Folder: ", d)
            
            for i, f in enumerate(os.listdir(_dir)):

                if f == ".DS_Store":
                    continue

                img = cv2.imread(os.path.join(_dir, f))
                resized = cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join("test", d, f), resized)


def resize(height, width):
    resize_train(height, width)
    resize_test(height, width)


resize(100, 100)