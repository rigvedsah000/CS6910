# CS6910 ASSIGNMENT 1 / README

### Terminology
- accuracy => Training Accuracy
- loss => Training Loss
- val_accuracy => Validation Accuracy
- val_loss => Validation Accuracy
- test_accuracy => Testing Accuracy
- test_loss => Testing Loss
### Training and Testing Models
The model will automatically train on 54,000 train and 6,000 validation datapoints and will be tested on 10,000 testpoints.

##### 
To train and validate model on wandb for different hyperparameter configurations please go to **_wandb_run.py_** file and do the following steps,

- Change the **config** dictionary to include the parameter you are interested in running e.g.
```
"optimiser": {
        "values": ["adam", "rmsprop", "nadam"]
        },
        "batch_size": {
            "values": [64]
        },
        "init_strategy": {
            "values": ["random", "xavier"]
        }
```
- Setup sweep with a **Project Name** e.g.
```
sweep_id = wandb.sweep(sweep_config, project="Assignment 1")
wandb.agent(sweep_id, project="Assignment 1", function=main)
```
- By default the loss will be **Cross Entropy Loss** but if you want to also compute **Squared Error Loss** then change the variable **_loss_functions_** in **main** function as,
```
loss_functions = [ "cross_entropy", "sq_loss" ]
```
This will generate squared error accuracy and loss plots in wandb along with cross entropy losses. However CE loss will be logged at every epoch whereas SE loss will only be logged only once.
- Run the file using command line as
```
python wandb_run.py
 ```
 ##### Note:
 If you are running a single model (each hyperparameter in config file has a single value) the please add **count = 1** as shown otherwise wandb will keep running the same model again and again i.e.
```
wandb.agent(sweep_id, project="Assignment 1", function=main, count=1)
```
 ### Adding a new Optimizer
 - Create a new file by name **_new_optimizer.py_**.
 - Create a function by the name **new_optimizer**, add required parameters and logic according to the need and use existing files and functions for forward and back propagation, calculating accuracy, creating confusion matrix etc.
 - Add following line to **_train.py_** i.e.
```
elif optimiser == "new_optimizer":
    return new_optimizer.new_optimizer(train_x, train_y, val_x, val_y, d, hl, ol, ac, lf, epochs, eta, init_strategy, batch_size)
```
- Add following to **config** variable in **_wandb_run.py_** file to call your optimiser from wandb,
```
"optimiser": { "values": ["adam", "my_optimizer"] },
```
 