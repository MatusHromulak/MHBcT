# MHBcT
#MNIST_CNN.py
Run as:
> python MNIST_CNN.py

Dependencies:
TensorFlow
Keras
TensorBoard

Supported arguments:
--aug_e		- enable data augmentation
--board_se	- enable TensorBoard logging
--mod_se	- export model data
--hist_se	- export trainning history data
--hyp_se	- export hyperparameters
--arch_se	- export model architecture to json

It is recommended to redirect console output to a file:
> python MNIST_CNN.py console_out.txt 2>&1

If you want to use TensorBoard:
- before running the program, open TensorBoard in a separate command window:
> tensorboard --logdir=C:\...\MNIST_result
- if you want to model multiple runs, put the parent directory as the --logdir arguments
- open the TensorBoard dashboard in a browser:
localhost:6006