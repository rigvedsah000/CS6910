import cv2
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np 
import time
from os.path import isfile, join
import sys


#Load yolo
#Load yolo
def load_yolo3():
    print("function invoked: load_yolo3")
    model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    class_type = []
    with open("coco.names", "r") as f:
        class_type = [line.strip() for line in f.readlines()]
    layers_names = model.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in model.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(class_type), 3))
    return model, class_type, colors, output_layers


def detect_objects(img, net, outputLayers):
    print("function invoked: detect_objects")
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs



def get_box_dimensions(outputs, height, width):
    print("function invoked: get_box_dimensions")
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids



def draw_labels(counter,boxes, confs, colors, class_ids, classes, img): 
    print("function invoked: draw labels")
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imwrite("frames/captioned_image"+str(counter)+".jpg", img)
    
    
    
def start_video(video_path):
    print("function invoked:",start_video)
    model, classes, colors, output_layers = load_yolo3()
    cap = cv2.VideoCapture(video_path)
    counter = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==False:
            return
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(counter ,boxes, confs, colors, class_ids, classes, frame)
        counter += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    
    
    
    
def frames_to_video(vid_path,image_name,fps):
    frame_list = []

    for i in range(len(image_name)):
        photo=image_name[i]
        img_array = cv2.imread(photo)
        frame_list.append(img_array)
        
        height, width, depth = img_array.shape
        dimensions = (width,height)
    
    result = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, dimensions)

    for i in range(len(frame_list)):
        result.write(frame_list[i])
    result.release()
    
    

    
    
# Create folder if not exists
if (os.path.exists("frames")):
    directory = 1
else:
    os.makedirs("frames")
    
# To clean the folder for storing the frames     
for files in os.listdir("frames"):
    path = os.path.join("frames", files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)

        
video_path = "videos/Cows on Highways.mp4"
start_video(video_path)  
cv2.destroyAllWindows() 
    
    
    
folder = "frames/"
n_files = 0
for files in os.listdir(folder):
    n_files += 1
print("Total number of frames in the video :",n_files)
    
 
image_name = []
for frame_num in range(0,n_files):
    photo = folder+"captioned_image"+str(frame_num)+".jpg"
    image_name.append(photo)
    

# Adjust frame rate of the video
fps = 30
caption_video = 'captioned_video.mp4'

# call to function converting the frames into video
frames_to_video(caption_video , image_name, fps)
print("video is successfully captioned. Enjoy it .")


sys.exit()