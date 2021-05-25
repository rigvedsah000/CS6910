# CS6910 ASSIGNMENT 3 / README

### Terminology
- accuracy => Training Accuracy
- loss => Training Loss
- val_accuracy => Validation Accuracy
- test_accuracy => Testing Accuracy


### Step 1 : 
Download all the files present in the github under Assignment 3 folder and keep all these in one folder.
### Step 2 :
Put "dakshina_dataset_v1.0" dataset in the same folder. 

(We have used Romanized word to Devanagri word conversion)

### Case 1: Model (without attention mechanism)

 If you want to run the model (without attention mechanism), then simply open **wandb_run.py** python file and do the follow steps : 
- Change the **sweep_config** dictionary to include the parameter you are interested in running 
```
"parameters": {
        "embedding_size": {                         # Size of Embedding
            "values": [256]          
        },
        "encoder_layers": {                         # Number of encoder layers
            "values": [3]
        },
        "decoder_layers": {                         # Number of deoder layers
            "values": [3]
        },
        "cell_type": {                              # RNN/LSTM/GRU
            "values": ["gru"]
        },
        "dropout": {                                # Value of Dropout in cell layer
            "values": [0.3]                               
        },
        "beam_size": {                              # Size of beam width
            "values": [4]
        }
    }
```
- Setup sweep with a **Project Name** e.g.
```
sweep_id = wandb.sweep(sweep_config, project="Assignment 3")
wandb.agent(sweep_id, project="Assignment 3", function=main)
```
- Run the file  **wandb_run.py**  . 
Now wandb automatically logs the accuracy (training accuracy character wise) , loss, val_accuracy, parallel coordinates plots, correlation table. 



### Case 2: Model (with attention mechanism) : 
If you want to run the model (with attention mechanism) , then simply run **wandb_run_attention.py**  python file :
- Change the **sweep_config** dictionary to include the parameter you are interested in running 
```
"parameters": {
        "embedding_size": {                         # Size of Embedding
            "values": [256]          
        },
        "encoder_layers": {                         # Do not change it. It is a single 
                                                      layered attention model
            "values": [1]
        },
        "decoder_layers": {                         # Do not change it. It is a single 
                                                      layered attention model
            "values": [1]
        },
        "cell_type": {                              # RNN/LSTM/GRU
            "values": ["gru"]
        },
        "dropout": {                                # Value of Dropout in cell layer
            "values": [0.3]                               
        },
        "beam_size": {                              # Size of beam width
            "values": [0]
        }
    }
```
- Setup sweep with a **Project Name** e.g.
```
sweep_id = wandb.sweep(sweep_config, project="Assignment 3")
wandb.agent(sweep_id, project="Assignment 3", function=main)
```
- Run the file  **wandb_run_attention.py**
 
Now wandb automatically logs the accuracy (training accuracy character wise) , training loss, val_accuracy, parallel coordinates plots, correlation table. 
Also, wandb will log the heatmap showing the dependency of output character on the input characters.


### Case 3: Visualisation  : 
For visualisation, you need a model. For this, you need to run the "wandb_run_attention.py". Then some data for visualisation will be stored in the text files.
We have provided colab notebook ("DL Assignment3 Visualization.ipynb") for better visualisation.
Just open that.

We have implemented two types of visualisation . These are as:
##### Visualizing (When the model is decoding the i-th character in the output, which is the input character that it is looking at ? ) :

1. When you run the "wandb_run_attention.py", then a text file with name "conn_vis.txt" would have been generated. Upload that on the google colab.
2. Now, go to second last line of the first cell of the notebook. 
3. Set the number of input words "visualize_connectivity(10) function call" that you would like to visualize. [ We have set by default as 10]
4. Comment last  line and uncomment second last line. Put the argument of second last line as number of words you want to visualize.
 ```
visualize_connectivity(10)
# visualize_lstm(10, 0)
```
5. Run the first cell of the notebook.

The visualisation of the words will be shown in the output. There we can visualise that when the model is decoding the i-th character in tht output, then which input character it is looking at.

##### Visualizing for what character a particular neuron in a layer is activated. :
1. Open the provided colab notebook ("DL Assignment3 Visualization.ipynb")
2. When you run the "wandb_run_attention.py", then some text file with name  for example : "lstm_vis_0.txt" would have been generated. This is corresponding to the first input word. Similarly, there would have files corresponding to other word. Suppose you want to visualise cell on first word. Then upload "lstm_vis_0.txt" that on the google colab. 
2. Now,  go to last line of first cell. 
3. Set the argument of visualize_lstm(1, 0). The first argument is number of word you want to visualise. The second argument is the cell number on which you want to visualise. [ By default , it is set as 10, 0]
4. Uncomment last line and comment out the second last line.
```
# visualize_connectivity(10)
 visualize_lstm(1, 0)                # 200 is the cell number to visualise
```
5. Run the first cell of the notebook.

The visualisation of the words will be shown in the output. There we can visualise that for what character that cell will be activated.

