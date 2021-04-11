import os, os.path, cv2, random, numpy as np

def load_train_images(path, sampling_factor = 1):

    data = []
   
    for d in os.listdir(path):
        _dir = os.path.join(path, d)
    
        print("Processing Folder: ", d)

        files = os.listdir(_dir)
        
        sample_count, size = 0, len(os.listdir(_dir))
        a = [0] * size

        while(sample_count != int(size * sampling_factor)):
            
            i = random.randrange(0, size)

            if a[i] == 0:
                data.append(np.array(cv2.imread(os.path.join(_dir, files[i]))))
                a[i] = 1
                sample_count += 1

    return np.array(data)