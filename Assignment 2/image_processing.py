import os, os.path, cv2, random
from math import ceil

def resize_train(height, width):
    path = "inaturalist_12K/train"

    os.mkdir(os.path.join("train"))
    os.mkdir(os.path.join("val"))

    for d in os.listdir(path):
        
        _dir = os.path.join(path, d)
    
        if os.path.isdir(_dir):

            os.mkdir(os.path.join("train", d))
            os.mkdir(os.path.join("val", d))
            
            print("Processing Folder: ", d)
            
            # Test Image ==> 0
            #  Val Image ==> 1

            files = os.listdir(_dir)
            size = len(files)
            a = [0] * size

            val_count = 0
            
            while(val_count != ceil(0.1 * size)):
                i = random.randrange(0, size)
                
                if a[i] == 0:
                    a[i] = 1
                    val_count += 1

            for i in range(len(a)):

                f = files[i]

                if f == ".DS_Store":
                    continue

                img = cv2.imread(os.path.join(_dir, f))
                resized = cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA)

                if a[i] == 0:
                    cv2.imwrite(os.path.join("train", d, f), resized)
                else:
                    cv2.imwrite(os.path.join("val", d, f), resized)

def resize_test(height, width):
    path = "inaturalist_12K/val"

    os.mkdir(os.path.join("test"))

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

resize(250, 250)