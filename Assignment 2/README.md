# CS6910 ASSIGNMENT 2 / README

### Terminology
- accuracy => Training Accuracy
- loss => Training Loss
- val_accuracy => Validation Accuracy
- val_loss => Validation Accuracy
- test_accuracy => Testing Accuracy


### Step 1 : 
Download all the files present in the github under Assignment 2 folder and keep all these in one folder.
### Step 2 :
Put "inaturalist_12K" dataset in the same folder. 
### Step 3 :
Case 1 : If you want to work on image size of 100 * 100 , then simply run **image_processing.py**  python file .  By default , the size is set to 100*100 . 
Case 2 : If you want to work on image size other than 100 * 100 , then just open the **image_processing.py** file and go to the last line  and change the image size to whatever you want. :
```
resize(100, 100)
```
This step 3 will form 3 different folder "train","test" and "val" in the main directory.

### Step 4  : 
Execute the instructions according to the Part you want to execute:
#### Part A :
##### Case 1 : 
If you want to visualise the accuracy , loss, predictions on sample test images etc in wandb , then continue with this part.
You need to open  **Part A/wandb_run.py** file and do the follow steps :
- Change the **sweep_config** dictionary to include the parameter you are interested in running 
```
"parameters": {
        "filters": {
            "values": [64]          # Number of filters to keep in first layer
        },
        "organisation" :{
            "values" : ["double"]  #" same": all layer have same number of filters
                                #"double":number of filters doubles subsequent layers
                                #"half": number of filter get halve subsequent layers
            },
        "augmentation": {
            "values": ["yes"]                   # Yes for augmentation otherwise no
        },
        "dropout": {
            "values": [0.2]                                    # value of dropout 
        },
        "batch_normalization": {
            "values": ["yes"]
        }
    }
```
- Setup sweep with a **Project Name** e.g.
```
sweep_id = wandb.sweep(sweep_config, project="Assignment 2")
wandb.agent(sweep_id, project="Assignment 2", function=main)
```
- Run the file  **wandb_run.py**  . 
Now wandb automatically logs the accuracy , loss, val_acc , val_loss  . 


Wandb  will automatically logs the accuracy plots, loss plots , prediction on sample test images etc. 

##### Case 2 : 
If you want to visualise filter or visualise Guided Backpropagation ,then you need a model . In this case,  you have to run the Case 1 exccept that one change . Just comment out line number 132. This will save the model trained. Comment out the below line in the **wandb_run.py**
```
#     model.save("my_model.h5")
```
- Now for Visualising Filter, just run "filter_visualisation.py"
- Or , For Visualising Guided Backpropagation, run "Guided Backpropagation.py"
In both the cases, the plot visualising them will be shown and also they will be saved in Part A folder.

##### Note :
Above cases in part A were executed with wandb with bydefault run count as 1. So, they will only run for one time and then the sweeping procedure will stop. If you wanted to run the wandb sweep multiple times , then just make the following change to last line in the **wandb_run.py**.
By default (for one run):
```
wandb.agent(sweep_id, project="Assignment 2", function=main, count = 1)
```
For multiple runs:
```
wandb.agent(sweep_id, project="Assignment 2", function=main)
```

#### Part B :
This part is related to  use of already existing pretrained model on our dataset .
You need to open  **wandb_run.py** file and do the follow steps :
- Change the **sweep_config** dictionary to include the parameter you are interested in running . 
**model** : Pretrained_model you want to use . 
**trainable_config_layers** : Number of layers of pre trained model you want to train 
-1  : All layers will be trained
 0  : No layer of pretrained model will be trained
 1  : Layers from the last convolutional layer upto the output layer will be trained
 2  : Layers from the second last convolutional layer upto the output layer will be trained
Now, change the configuration  according to you.
```
"parameters": {
        "model": {
            "values": ["inception_v3", "inception_resnet_v2", "resnet_50", "xception"]
        },
        "trainable_conv_layers" : {
            "values" : [0, 1, 2, -1]               
        }
}
```
- Setup sweep with a **Project Name** e.g.
```
sweep_id = wandb.sweep(sweep_config, project="Assignment 2")
wandb.agent(sweep_id, project="Assignment 2", function=main)
```
- Run the file  **wandb_run.py**  . 
Now wandb automatically logs the accuracy , loss, val_acc , val_loss  . 
#### Part C :
##### Case 1 : 
If you want to execute code or want to convert your own video to captioned version, then follow this case :
You need to download weights of YoloV3 model. You can download weights of model from here :
```
YoloV3 weights link : https://pjreddie.com/darknet/yolo/
(Download only YoloV3 weights.  cfg file is already present in folder, so no need to download cfg file.)
```
Now, if the downloaded Yolo V3 weights file is by name "yolov3.weights", then good, otherwise rename it to "yolov3.weights" and put it in Part C folder.
Now, if you want to make your video captioned , then put your video in "Part C/videos" .
Now , open **main.py** file inside Part C . Go to line number 132.
```
Change   "videos/Cows on Highways.mp4"   to   "videos/your_video_name.mp4"
```
Now run  **main.py**  file.  Your video will be converted into captioned video and will be saved by name  **captioned_video.mp4** in Part C folder.


##### Case 2 : 
If you just want to view application/video of Object Detection using Yolo V3 , then simply go to below youtube link .
```
Youtube video link : https://www.youtube.com/watch?v=w1JvCeC6ZGk
```
