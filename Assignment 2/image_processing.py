import os, os.path, cv2, random
from math import ceil
import os , shutil

def resize_train(height, width):
    path = "inaturalist_12K/train"
    
    # Create folder if not exists
    if (os.path.exists("train")):
        directory = 1
    else:
        os.makedirs("train")


    if (os.path.exists("val")):
        directory = 1
    else:
        os.makedirs("val")
        
     
    # To clean the folder for storing the files 
    for files in os.listdir("train"):
        loc = os.path.join("train", files)
        try:
            shutil.rmtree(loc)
        except OSError:
            os.remove(loc) 

    for files in os.listdir("val"):
        loc = os.path.join("val", files)
        try:
            shutil.rmtree(loc)
        except OSError:
            os.remove(loc) 


    # os.mkdir(os.path.join("train"))
    # os.mkdir(os.path.join("val"))

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

    if (os.path.exists("test")):
        directory = 1
    else:
        os.makedirs("test")


    for files in os.listdir("test"):
        loc = os.path.join("test", files)
        try:
            shutil.rmtree(loc)
        except OSError:
            os.remove(loc)
            
    # os.mkdir(os.path.join("test"))

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